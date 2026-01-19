import torch
import sys
import os

# Add project path
sys.path.insert(0, os.getcwd())

# Import the modified module
try:
    from ultralytics.nn.smoke_modules import ECPConv, ECPConvBlock, EMA
    print("[OK] Modules imported successfully.")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    exit(1)

# Test ECPConv
print("\n--- Testing ECPConv ---")
x = torch.randn(2, 64, 32, 32)  # Batch=2, Channels=64, H=W=32

# Test 1: Same channels, stride=1
ecp1 = ECPConv(64, 64, k=3, s=1)
y1 = ecp1(x)
print(f"ECPConv(64->64, s=1): Input {x.shape} -> Output {y1.shape}")
assert y1.shape == (2, 64, 32, 32), f"Shape mismatch! Expected (2, 64, 32, 32), got {y1.shape}"
print("[PASS] ECPConv stride=1 works!")

# Test 2: Different channels, stride=2
ecp2 = ECPConv(64, 128, k=3, s=2)
y2 = ecp2(x)
print(f"ECPConv(64->128, s=2): Input {x.shape} -> Output {y2.shape}")
assert y2.shape == (2, 128, 16, 16), f"Shape mismatch! Expected (2, 128, 16, 16), got {y2.shape}"
print("[PASS] ECPConv stride=2 with channel change works!")

# Test ECPConvBlock
print("\n--- Testing ECPConvBlock ---")
block = ECPConvBlock(64, 128, s=2)
y_block = block(x)
print(f"ECPConvBlock(64->128, s=2): Input {x.shape} -> Output {y_block.shape}")
assert y_block.shape == (2, 128, 16, 16), f"Shape mismatch! Expected (2, 128, 16, 16), got {y_block.shape}"
print("[PASS] ECPConvBlock works!")

# Test gradient flow
print("\n--- Testing Gradient Flow ---")
x_grad = torch.randn(2, 64, 32, 32, requires_grad=True)
block_grad = ECPConvBlock(64, 64, s=1)
y_grad = block_grad(x_grad)
loss = y_grad.sum()
loss.backward()
print(f"Gradient on input: {x_grad.grad is not None}")
assert x_grad.grad is not None, "Gradient not flowing!"
print("[PASS] Gradients flow correctly!")

print("\n" + "="*50)
print("ALL TESTS PASSED! ECPConv is correctly implemented.")
print("="*50)
