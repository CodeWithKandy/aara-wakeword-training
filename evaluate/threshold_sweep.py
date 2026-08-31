"""Test the trained model directly against precomputed features — with threshold analysis."""
import numpy as np
import onnxruntime as ort

pos_features = np.load("training_output/hey_aara/positive_features_test.npy").astype(np.float32)
neg_features = np.load("training_output/hey_aara/negative_features_test.npy").astype(np.float32)

sess = ort.InferenceSession("model/hey_aara.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name


def score_all(features):
    scores = []
    for row in features:
        out = sess.run(None, {input_name: row[None, :, :]})[0]
        scores.append(float(out.squeeze()))
    return np.array(scores)


pos_scores = score_all(pos_features)
neg_scores = score_all(neg_features)

print(f"Positive: mean {pos_scores.mean():.4f}, max {pos_scores.max():.4f}, min {pos_scores.min():.4f}")
print(f"Negative: mean {neg_scores.mean():.4f}, max {neg_scores.max():.4f}, min {neg_scores.min():.4f}")
print()
for threshold in [0.3, 0.5, 0.7, 0.9]:
    recall = (pos_scores >= threshold).mean() * 100
    false_accept = (neg_scores >= threshold).mean() * 100
    print(f"Threshold {threshold}: recall={recall:.1f}%  false-accept={false_accept:.1f}%")