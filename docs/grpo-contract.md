# One-step GRPO contract

For one `click-test` task instance, the optimized sequence is:

```text
[prompt tokens, not trainable] [completion tokens, trainable] [padding, not trainable]
```

The prompt contains the system instruction, goal, accessibility tree, and final
`Action:` marker. The completion is exactly what the behavior policy sampled,
including EOS when generated. Environment observations and parsed BrowserGym
actions are metadata; they are never appended and labelled as policy-generated
tokens.

Each completion stores its behavior-policy log probability at every completion
position. `completion_ids`, `behavior_logprobs`, and `trainable_mask` therefore
have identical lengths. Padding may occupy completion storage but is masked out.

All members of a GRPO group share the same `task_id` and exact prompt token IDs.
Rewards are standardized within that group using sample standard deviation plus
`1e-4`, matching TRL 0.29.0's `scale_rewards="group"` implementation. An
equal-reward group has zero advantages and consequently no policy-gradient signal.

An invalid model string is still sampled policy behavior. It remains in the
group, retains its token IDs and log probabilities, and receives reward zero. It
is not rewritten as `noop()` and is not dropped. The collector may avoid sending
invalid syntax to BrowserGym, but that execution choice does not alter the sampled
completion.

For the one-step collector, a successful environment termination has
`terminal=True`; a valid action that leaves the environment active exhausts the
imposed one-action budget and has `truncated=True`. Invalid syntax ends the rollout
as a terminal policy failure without stepping the environment.

`clipped_policy_loss` is a small dependency-free reference, not the production
trainer. For each trainable token it computes
`min(ratio * advantage, clipped_ratio * advantage)` and returns the negative mean.
Its purpose is to catch sign, clipping, masking, and old/new log-probability
alignment errors before integrating a trainer.

The first trainer integration pins `trl==0.29.0` and explicitly sets
`scale_rewards="group"`, `loss_type="grpo"`, `num_iterations=1`, and `beta=0.0`.
These choices match this small reference objective and avoid silently inheriting
future defaults. DAPO-style batch-token normalization is a later controlled
comparison, not part of this contract check.

`tests/test_trl_contract.py` exercises TRL 0.29.0's real `_compute_loss` method
with deterministic tensors. It verifies the clipped masked loss against the
dependency-free calculation and confirms that a fully masked truncated completion
contributes zero. The test is skipped unless the optional training dependencies
are installed.
