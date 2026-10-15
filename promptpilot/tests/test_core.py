import unittest
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from promptpilot.core import (
    DELETE,
    MOVE,
    GRAPH_STATE_DIM,
    STRICT_STATE_DIM,
    RESTORE,
    DistanceGraphEnv,
    GraphSample,
    PromptPilotConfig,
    QNetwork,
    greedy_optimize_prompts,
    optimize_prompts,
)
from promptpilot.improved import (
    CANDIDATE_DIM,
    STOP,
    CandidateActionEnv,
    CandidateQNetwork,
    optimize_prompts_improved,
)
from promptpilot.evaluate import (
    refine_mask,
    score_guidance,
    select_mask_index,
    select_sample_paths,
    use_candidate,
)
from promptpilot.reference_selector import (
    ReferenceFeatureClassifier,
    select_reference_prompts,
)
from promptpilot.train_improved import Transition, double_dqn_loss
from promptpilot.train import ReplayTransition, dqn_loss


def sample() -> GraphSample:
    generator = torch.Generator().manual_seed(7)
    return GraphSample(
        name="synthetic",
        features=torch.randn(64, 12, generator=generator),
        positive_indices=torch.arange(0, 24),
        negative_indices=torch.arange(32, 56),
        image_size=112,
        patch_size=14,
    )


def reference_sample(tmp_path: str) -> GraphSample:
    item = sample()
    item.reference_features = item.features.clone()
    mask = np.zeros((112, 112), dtype=np.uint8)
    mask[:56, :56] = 255
    Image.fromarray(mask).save(tmp_path)
    item.metadata = {"reference_mask": tmp_path}
    return item


class CoreTests(unittest.TestCase):
    def test_reference_selector_uses_reference_features_without_target_mask(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as handle:
            item = reference_sample(handle.name)
            pos, neg, trace = select_reference_prompts(
                item,
                positive_count=4,
                negative_count=6,
                min_separation=1.0,
                candidate_source="initial",
            )
        self.assertEqual(len(pos), 4)
        self.assertEqual(len(neg), 6)
        self.assertFalse(torch.isin(pos, neg).any())
        self.assertEqual(trace["counts"], [4, 6])

    def test_reference_selector_supports_trained_patch_classifier(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as handle:
            item = reference_sample(handle.name)
            classifier = ReferenceFeatureClassifier(item.features.shape[1], hidden_dim=8)
            pos, neg, trace = select_reference_prompts(
                item,
                positive_count=3,
                negative_count=5,
                min_separation=1.0,
                reference_classifier=classifier,
                score_mode="hybrid-linear",
            )
        self.assertEqual((len(pos), len(neg)), (3, 5))
        self.assertFalse(torch.isin(pos, neg).any())
        self.assertEqual(trace["score_mode"], "hybrid-linear")

    def test_sam_quality_gate_requires_the_configured_improvement(self):
        self.assertTrue(use_candidate("none", 0.9, 0.1, 0.0))
        self.assertTrue(use_candidate("sam-predicted-iou", 0.7, 0.71, 0.01))
        self.assertFalse(use_candidate("sam-predicted-iou", 0.7, 0.709, 0.01))
        self.assertFalse(use_candidate("sam-predicted-iou", 0.7, float("nan"), 0.0))

    def test_score_guidance_produces_sam_box_and_logits(self):
        scores = torch.zeros(64)
        scores[18:22] = 2.0
        box, logits = score_guidance(
            scores,
            positive_indices=torch.tensor([18]),
            grid_size=8,
            image_size=112,
            quantile=0.7,
            scale=2.0,
        )
        self.assertEqual(box.shape, (4,))
        self.assertEqual(logits.shape, (1, 256, 256))
        self.assertTrue(np.isfinite(logits).all())

    def test_dino_mask_selection_can_override_sam_quality(self):
        masks = np.zeros((2, 8, 8), dtype=bool)
        masks[0, :4] = True
        masks[1, 4:] = True
        scores = torch.cat((torch.ones(8), -torch.ones(8)))
        selected = select_mask_index(
            masks,
            quality=np.asarray([0.1, 0.9]),
            mode="dino-contrast",
            patch_scores=scores,
        )
        self.assertEqual(selected, 0)

    def test_sample_list_preserves_requested_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("alpha", "beta"):
                bundle = root / name
                bundle.mkdir()
                for filename in ("features.pt", "initial_indices_pos.pt", "initial_indices_neg.pt"):
                    (bundle / filename).touch()
            sample_list = root / "samples.txt"
            sample_list.write_text("beta\nalpha\n", encoding="utf-8")
            selected = select_sample_paths(str(root), str(sample_list))
        self.assertEqual([path.name for path in selected], ["beta", "alpha"])

    def test_network_matches_architecture(self):
        network = QNetwork(STRICT_STATE_DIM)
        self.assertEqual(tuple(network(torch.zeros(STRICT_STATE_DIM)).shape), (3,))
        self.assertEqual(network.layers[0].out_features, 1024)
        self.assertEqual(network.layers[2].out_features, 64)

    def test_state_uses_only_equations_3_to_7(self):
        config = PromptPilotConfig(
            max_steps=20,
            state_encoder="strict-five",
            distance_normalization="official-minmax",
        )
        env = DistanceGraphEnv(sample(), config, seed=1)
        self.assertEqual(tuple(env.state_vector().shape), (STRICT_STATE_DIM,))
        for _ in range(10):
            state, _, _, _ = env.step(RESTORE)
        self.assertEqual(tuple(state.shape), (STRICT_STATE_DIM,))
        self.assertTrue(torch.isfinite(state).all())

    def test_graph_state_encodes_dual_space_relation_distributions(self):
        config = PromptPilotConfig(
            max_steps=20,
            state_encoder="graph-moments",
            distance_normalization="official-minmax",
        )
        env = DistanceGraphEnv(sample(), config, seed=1)
        for _ in range(10):
            state, _, _, _ = env.step(RESTORE)
        self.assertEqual(tuple(state.shape), (GRAPH_STATE_DIM,))
        self.assertTrue(torch.isfinite(state).all())
        self.assertTrue(torch.allclose(state[::4][:5], torch.from_numpy(env._stats)))

    def test_action_constraints_and_counts(self):
        env = DistanceGraphEnv(sample(), PromptPilotConfig(max_steps=20), seed=1)
        self.assertEqual(env.valid_actions().tolist(), [True, False, False])
        for _ in range(10):
            env.step(RESTORE)
        self.assertGreaterEqual(env.counts()[0], 5)
        self.assertGreaterEqual(env.counts()[1], 5)
        self.assertEqual(env.valid_actions().tolist(), [True, True, True])
        before = sum(env.counts())
        env.step(DELETE)
        self.assertEqual(sum(env.counts()), before - 1)

    def test_reward_is_equation_8_potential_difference(self):
        config = PromptPilotConfig(max_steps=20, alpha=5.0, beta=1.0)
        env = DistanceGraphEnv(sample(), config, seed=3)
        previous = env._stats.copy()
        _, reward, _, info = env.step(RESTORE)
        expected = env.potential(info["statistics"]) - env.potential(previous)
        self.assertAlmostEqual(reward, expected, places=7)

    def test_seed_makes_transition_reproducible(self):
        config = PromptPilotConfig(max_steps=20)
        left = DistanceGraphEnv(sample(), config, seed=99)
        right = DistanceGraphEnv(sample(), config, seed=99)
        for _ in range(12):
            action = RESTORE if bool(left.valid_actions()[RESTORE]) else MOVE
            left.step(action)
            right.step(action)
        self.assertEqual(left.active, right.active)

    def test_policy_rollouts_select_only_by_equation_8(self):
        network = QNetwork(STRICT_STATE_DIM)
        pos, neg, trace = optimize_prompts(
            network,
            sample(),
            PromptPilotConfig(max_steps=20, state_encoder="strict-five"),
            seed=10,
            rollouts=4,
        )
        self.assertEqual(trace["rollouts"], 4)
        self.assertEqual(trace["final_potential"], max(trace["rollout_potentials"]))
        self.assertEqual(set(trace["action_counts"]), {"restore", "delete", "move"})
        self.assertGreaterEqual(len(pos), 5)
        self.assertGreaterEqual(len(neg), 5)

    def test_policy_can_retain_published_terminal_epsilon(self):
        network = QNetwork(STRICT_STATE_DIM)
        _, _, trace = optimize_prompts(
            network,
            sample(),
            PromptPilotConfig(max_steps=100, state_encoder="strict-five"),
            seed=10,
            rollouts=8,
            inference_epsilon=0.1,
        )
        self.assertEqual(trace["inference_epsilon"], 0.1)
        self.assertGreater(trace["feasible_rollouts"], 0)
        self.assertTrue(
            any(counts["move"] > 0 for counts in trace["rollout_action_counts"])
        )

    def test_dqn_loss_uses_three_action_equation_9_target(self):
        online = QNetwork(STRICT_STATE_DIM)
        target = QNetwork(STRICT_STATE_DIM)
        target.load_state_dict(online.state_dict())
        transition = ReplayTransition(
            state=torch.zeros(STRICT_STATE_DIM),
            action=RESTORE,
            reward=0.25,
            next_state=torch.ones(STRICT_STATE_DIM),
            next_valid=torch.tensor([True, False, False]),
            done=False,
        )
        loss = dqn_loss(
            online,
            target,
            [transition],
            gamma=0.1,
            device=torch.device("cpu"),
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in online.parameters()))

    def test_public_mask_refinement_keeps_large_components(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[2:12, 2:12] = 1
        mask[20:22, 20:22] = 1
        refined = refine_mask(mask, threshold=0.5)
        self.assertTrue(refined[5, 5])
        self.assertFalse(refined[20, 20])

    def test_greedy_optimizer_respects_bounds_and_improves_potential(self):
        pos, neg, trace = greedy_optimize_prompts(sample(), PromptPilotConfig())
        self.assertGreaterEqual(len(pos), 5)
        self.assertGreaterEqual(len(neg), 5)
        self.assertLessEqual(len(pos), 20)
        self.assertLessEqual(len(neg), 20)
        self.assertGreaterEqual(trace["final_potential"], trace["initial_potential"] - 1e-7)

    def test_greedy_optimizer_can_use_release_weight_node_counts(self):
        pos, neg, _ = greedy_optimize_prompts(sample(), PromptPilotConfig(), target_counts=(6, 12))
        self.assertEqual(len(pos), 6)
        self.assertEqual(len(neg), 12)

    def test_improved_environment_exposes_concrete_node_actions(self):
        env = CandidateActionEnv(sample(), PromptPilotConfig(max_steps=20), seed=3)
        while min(env.counts()) < env.config.min_nodes:
            action = env.candidate_actions()[0]
            self.assertEqual(action.operation, RESTORE)
            env.step_candidate(action)

        actions = env.candidate_actions()
        descriptors = env.candidate_descriptors(actions)
        self.assertEqual(tuple(descriptors.shape), (len(actions), CANDIDATE_DIM))
        self.assertTrue(torch.isfinite(descriptors).all())
        self.assertIn(STOP, [action.operation for action in actions])
        self.assertNotIn(DELETE, [action.operation for action in actions])

    def test_low_count_initial_state_can_move_or_stop_when_restore_is_impossible(self):
        low_count = sample()
        low_count.positive_indices = torch.arange(0, 2)
        env = CandidateActionEnv(low_count, PromptPilotConfig(max_steps=20), seed=3)
        env.reset_to_initial_prompts()
        actions = env.candidate_actions()
        self.assertTrue(actions)
        self.assertIn(STOP, [action.operation for action in actions])

    def test_improved_action_changes_the_requested_node(self):
        left = CandidateActionEnv(sample(), PromptPilotConfig(max_steps=20), seed=4)
        right = CandidateActionEnv(sample(), PromptPilotConfig(max_steps=20), seed=999)
        action = left.candidate_actions()[7]
        self.assertEqual(action, right.candidate_actions()[7])
        left.step_candidate(action)
        right.step_candidate(action)
        self.assertEqual(left.active, right.active)

    def test_improved_inference_starts_from_initial_prompts(self):
        env = CandidateActionEnv(sample(), PromptPilotConfig(max_steps=20), seed=4)
        env.reset_to_initial_prompts()
        self.assertEqual(env.counts(), (20, 20))
        self.assertEqual(len(env.removed), 8)
        env.reset_to_initial_prompts(prune_to_bounds=False)
        self.assertEqual(env.counts(), (24, 24))
        self.assertEqual(len(env.removed), 0)

    def test_candidate_network_and_double_dqn_loss(self):
        online = CandidateQNetwork()
        target = CandidateQNetwork()
        target.load_state_dict(online.state_dict())
        transition = Transition(
            descriptor=torch.zeros(CANDIDATE_DIM),
            reward=0.5,
            next_descriptors=torch.stack(
                (torch.zeros(CANDIDATE_DIM), torch.ones(CANDIDATE_DIM))
            ),
            done=False,
        )
        loss = double_dqn_loss(
            online,
            target,
            [transition],
            gamma=0.1,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(online(torch.zeros(5, CANDIDATE_DIM)).shape), (5,))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in online.parameters()))

    def test_improved_inference_returns_best_feasible_graph(self):
        network = CandidateQNetwork()
        pos, neg, trace = optimize_prompts_improved(
            network,
            sample(),
            PromptPilotConfig(max_steps=20),
            seed=8,
        )
        self.assertGreaterEqual(len(pos), 5)
        self.assertGreaterEqual(len(neg), 5)
        self.assertGreaterEqual(
            trace["best_potential"], trace["first_feasible_potential"] - 1e-7
        )


if __name__ == "__main__":
    unittest.main()
