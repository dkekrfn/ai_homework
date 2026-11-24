# -*- coding: utf-8 -*-
import torch
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

torch.manual_seed(0)

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

   # ========== 函数定义（初始化+训练循环）==========
def train(lr, epochs=2000):
    W1 = torch.randn(2, 2, requires_grad=True)    # 参数初始化
    b1 = torch.zeros(2, requires_grad=True)
    W2 = torch.randn(2, 1, requires_grad=True)
    b2 = torch.zeros(1, requires_grad=True)
    loss_history = [] 


    for epoch in range(epochs):
        h = torch.sigmoid(X @ W1 + b1)
        p = torch.sigmoid(h @ W2 + b2)
        loss = -(y*torch.log(p) + (1-y)*torch.log(1-p)).mean()
        loss.backward()
        loss_history.append(loss.item())

        with torch.no_grad():
            W1 -= lr * W1.grad
            b1 -= lr * b1.grad
            W2 -= lr * W2.grad
            b2 -= lr * b2.grad
        W1.grad.zero_(); b1.grad.zero_()
        W2.grad.zero_(); b2.grad.zero_()

    with torch.no_grad():
        h = torch.sigmoid(X @ W1 + b1)
        p = torch.sigmoid(h @ W2 + b2)
    return loss_history,h,p


# ---- 训练结束：打印判定表 ----
torch.manual_seed(0)
loss_history, h, p = train(0.5)

print("\nx1 x2 | target | hidden        | prob   | pred")
for i in range(4):
    pred = 1 if p[i] >= 0.5 else 0
    print(f"{int(X[i][0])}  {int(X[i][1])} |   {int(y[i][0])}    | "
          f"({h[i][0]:.3f},{h[i][1]:.3f}) | {p[i][0]:.4f} |  {pred}")
plt.figure(figsize=(6, 4))
plt.plot(loss_history)                 # 2000个点连成曲线
plt.xlabel("epoch"); plt.ylabel("loss")
plt.title("XOR 训练损失曲线 (lr=0.5)")
plt.savefig("loss_curve_xor.png", dpi=150)   # 存进 D:\ml_study
plt.show()                             # 弹窗显示

# ---- 损失曲线 + 隐藏层折叠图 ----

plt.figure(figsize=(5, 5))
hs = h.numpy()
for i, label in enumerate(["(0,0)", "(0,1)", "(1,0)", "(1,1)"]):
    plt.scatter(hs[i, 0], hs[i, 1], s=200, c="red" if y[i] == 1 else "blue")
    plt.annotate(label, (hs[i, 0], hs[i, 1]), fontsize=12)
plt.xlabel("h1"); plt.ylabel("h2")
plt.title("隐藏层空间：(0,0) 与 (1,1) 被折叠到同一点")
plt.savefig("hidden_space.png", dpi=150)
plt.show()

results = {}
for lr in [0.01, 0.1, 1.0, 10.0]:
    torch.manual_seed(0)
    hist, _, p_ = train(lr)
    results[lr] = hist
    n = sum((1 if p_[i] >= 0.5 else 0) == y[i] for i in range(4))
    print(f"lr={lr}: 最终loss={hist[-1]:.4f}, 判对{n}/4")


# ---- η 实验 ----
plt.figure(figsize=(7, 5))
for lr, hist in results.items():
    plt.plot(hist, label=f"lr={lr}")
plt.xlabel("epoch"); plt.ylabel("loss")
plt.title("学习率对训练的影响")
plt.legend()
plt.savefig("lr_compare.png", dpi=150)
plt.show()

