"""Real CPU GradScaler/AdamW checks for finite zero and invalid gradients."""

import unittest

from browser_control.lfm3_optimization import InvalidLoRAGradients, checked_lora_step

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch runtime is not installed")
class LoRAStepTests(unittest.TestCase):
    def setup_optimizer(self, *, unused=False, size=1):
        parameter = torch.nn.Parameter(torch.ones(size))
        parameters = {"lora_weight": parameter}
        if unused:
            parameters["lora_unused"] = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = torch.optim.AdamW(parameters.values(), lr=1e-3, weight_decay=0)
        scaler = torch.amp.GradScaler("cpu", init_scale=1.0)
        return parameter, parameters, optimizer, scaler

    def test_zero_gradient_keeps_adam_momentum_and_is_not_counted_as_signal(self):
        p, parameters, optimizer, scaler = self.setup_optimizer()
        scaler.scale((2 * p).sum()).backward()
        first = checked_lora_step(parameters, optimizer, scaler)
        self.assertTrue(first["nonzero_gradient"])
        self.assertEqual(first["unclipped_gradient_norm"], 2)
        before = p.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        scaler.scale((0 * p).sum()).backward()
        second = checked_lora_step(parameters, optimizer, scaler)
        self.assertFalse(second["nonzero_gradient"])
        self.assertEqual(second["unclipped_gradient_norm"], 0)
        self.assertEqual(second["scale"], 1)
        self.assertFalse(torch.equal(p, before))
        self.assertEqual(optimizer.state[p]["step"].item(), 2)

    def test_zero_without_prior_momentum_does_not_fabricate_weight_change(self):
        p, parameters, optimizer, scaler = self.setup_optimizer()
        before = p.detach().clone()
        scaler.scale((0 * p).sum()).backward()
        result = checked_lora_step(parameters, optimizer, scaler)
        self.assertFalse(result["nonzero_gradient"])
        self.assertTrue(torch.equal(p, before))

    def test_underflowed_norm_is_distinct_from_zero_gradient_entries(self):
        p, parameters, optimizer, scaler = self.setup_optimizer(size=2)
        scaler.scale((1e-30 * p).sum()).backward()
        result = checked_lora_step(parameters, optimizer, scaler)
        self.assertEqual(result["unclipped_gradient_norm"], 0)
        self.assertTrue(result["zero_gradient_norm"])
        self.assertTrue(result["nonzero_gradient"])

    def test_missing_gradient_stops_before_optimizer_changes_weights(self):
        p, parameters, optimizer, scaler = self.setup_optimizer(unused=True)
        before = p.detach().clone()
        scaler.scale(p.sum()).backward()
        with self.assertRaises(InvalidLoRAGradients) as error:
            checked_lora_step(parameters, optimizer, scaler)
        self.assertEqual(error.exception.names, ["lora_unused"])
        self.assertTrue(torch.equal(p, before))
        self.assertFalse(optimizer.state)

    def test_nan_and_infinite_gradients_still_stop_before_step(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                p, parameters, optimizer, scaler = self.setup_optimizer()
                before = p.detach().clone()
                scaler.scale(p.sum()).backward()
                p.grad.fill_(value)
                with self.assertRaises(InvalidLoRAGradients):
                    checked_lora_step(parameters, optimizer, scaler)
                self.assertTrue(torch.equal(p, before))
                self.assertFalse(optimizer.state)


if __name__ == "__main__":
    unittest.main()
