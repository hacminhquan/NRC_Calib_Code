# NRC-Cal: Predicting Probabilistic Calibratability via Neural Regression Collapse Geometry

Em suggest tên paper là **"Confidence Has a Shape. It's Already There"**

## Câu hỏi nghiên cứu cốt lõi

*Với một mô hình regression đã được train và đóng băng, độ lệch hình học giữa feature tầng cuối và cấu trúc Neural Regression Collapse lý tưởng — đo được chỉ qua một lượt forward pass, không cần train lại — có dự đoán được mức độ probabilistic miscalibration (PCE) hay không, và có thể dùng trực tiếp để sửa nó bằng một công thức closed-form hay không?*

Đây chính là phép chuyển ý tưởng NCCS (đã làm cho classification) sang regression, với đủ 3 mảnh ghép đã có sẵn — không phải nghĩ lại từ đầu:
- **Lý thuyết hình học:** Neural Regression Collapse (NeurIPS 2024) đã hình thức hóa NRC1-NRC3 cho regression, chứng minh chúng xuất hiện như nghiệm tối ưu khi regularization dương.
- **Công thức luận từng làm rồi:** NCTI/FaCe (đo NC-distance của model đóng băng → dự đoán property Y) — Y giờ đổi thành PCE thay vì transferability hay ECE.
- **Testbed có sẵn:** 57 dataset tabular regression + code công khai từ nghiên cứu QRT, đã có sẵn số PCE/NLL/CRPS baseline để đối chiếu ngay.

---

## Ba hướng tiếp cận nghiên cứu chính

### Hướng 1 — Correlational Study (bắt buộc làm trước, đây là go/no-go)

Tính NRC1 (collapse vào subspace n chiều, với n là số biến target), NRC2/NRC3 (tương tự equiangularity/self-duality nhưng chiếu lên subspace target thay vì simplex ETF) trên model BASE đã train sẵn từ QRT, với **nhánh Gaussian đơn** trước tiên (khớp đúng giả định lý thuyết NRC gốc, tránh nhiễu do mixture). Tương quan với PCE đã công bố sẵn của BASE trên từng dataset trong số 57 bộ.

- Nếu tương quan rõ → NRC-distance là một calibratability diagnostic thật sự cho regression, đúng như NCCS phía classification.
- Nếu không → vẫn là một phát hiện đáng công bố (negative result có kiểm chứng thực nghiệm rộng, hiếm khi ai làm).

### Hướng 2 — Correction: công thức closed-form sửa PCE dựa trên NRC

Nếu Hướng 1 dương tính, xây một phép biến đổi hậu kỳ dựa trực tiếp trên độ lệch NRC — tương tự cách NCCS dùng self-duality gap để định nghĩa $T_k$ cho từng lớp, ở đây định nghĩa một **calibration map được điều chỉnh theo mức độ NRC** thay vì ước lượng thuần túy từ PIT thực nghiệm như Kuleshov/QRT đang làm. Ý tưởng cụ thể: khi NRC-distance của một mẫu (hoặc một vùng feature) cao — tức feature của nó lệch xa subspace target lý tưởng — độ rộng của khoảng dự đoán nên được nới ra theo một hệ số phụ thuộc NRC-distance, trước khi áp Quantile Recalibration chuẩn. Đây là điểm khác biệt cốt lõi so với QRT: **QRT sửa bằng cách tích hợp bước hậu kỳ vào training (cần train lại)**; **NRC-Cal sửa hoàn toàn post-hoc, chỉ cần feature đã đóng băng** — đúng tinh thần training-free nhất quán với toàn bộ UCF Phase 1.

### Hướng 3 — Trục "n" (số biến target) tương tự trục K bên classification

Bài Generalized Neural Collapse (ICML 2024) đã chỉ ra khi K vượt quá chiều đặc trưng d, cấu trúc collapse cổ điển phải tổng quát hóa. Ở regression có một trục tương tự: **n** (số biến target liên tục, ví dụ n=1 cho hầu hết 57 dataset của QRT nhưng n có thể lớn hơn cho bài toán multivariate). Câu hỏi phụ đáng khai thác: liệu chất lượng dự đoán của NRC-distance có suy giảm khi n tăng, giống hệt cách hiệu quả của Vector/Dirichlet Calibration suy giảm khi K tăng? Đây là cách tận dụng lại đúng insight đã tìm cho classification (K-scaling) sang domain mới, tạo một mạch lý luận xuyên suốt hai bài báo.

---

## Thiết kế thực nghiệm cụ thể

| Bước | Nội dung | Nguồn tận dụng |
|---|---|---|
| 1. Model zoo | Lấy BASE (MLP dự đoán 1 Gaussian) từ 57 dataset QRT; nếu checkpoint không được phát hành thì train reproduction phải là một stage riêng | Code công khai của Dheur & Ben Taieb; checkpoint availability phải được kiểm chứng |
| 2. Tính NRC | NRC1/NRC2/NRC3 trên feature tầng cuối, một lượt forward qua tập calibration | Công thức từ NeurIPS 2024 + Geometric Analysis of NRC via Intrinsic Dimension |
| 3. Đối chiếu | Tương quan NRC-distance với PCE đã công bố sẵn, trước và sau QR chuẩn (Kuleshov) | Số liệu PCE/NLL/CRPS có sẵn trong Appendix G, I của QRT |
| 4. So sánh flexibility | Lặp lại bước 2-3 trên nhánh mixture-3-Gaussian và mixture-10-Gaussian trong Appendix A của QRT | Dùng checkpoint thật nếu được phát hành; nếu không, train reproduction với provenance đầy đủ |
| 5. Correction | Nếu bước 3 dương tính, cài công thức NRC-weighted recalibration map, so với QRC/QRTC/QREGC | Baseline có sẵn, chỉ cần thêm 1 phương pháp mới vào bảng so sánh |
| 6. Trục n | Tìm hoặc dựng thêm 2-3 dataset multivariate regression (n>1) để test độ suy giảm theo n | Có thể mượn dataset multivariate energy/weather đã nêu trong paper gốc NRC |

**Metric chính:** PCE (định nghĩa chuẩn theo QRT, M=100 mức quantile). **Metric phụ:** NLL, CRPS — để đảm bảo phương pháp mới không hy sinh sharpness để đổi lấy calibration, đúng tinh thần "calibration nhưng vẫn sharp" mà cả Kuleshov lẫn QRT đều nhấn mạnh.

---

## Định vị so với các công trình liên quan (Related Work)

| Công trình | Khác biệt với NRC-Cal |
|---|---|
| Kuleshov et al. 2018 (Quantile Recalibration) | Không dùng thông tin hình học feature — chỉ dựa trên PIT thực nghiệm, không biết *tại sao* model miscalibrated |
| QRT (AISTATS 2024) | Sửa bằng cách **tích hợp vào training** — cần train lại toàn bộ; NRC-Cal sửa **hoàn toàn post-hoc** |
| Neural Regression Collapse gốc (NeurIPS 2024) | Chỉ nối NRC với Test MSE — chưa từng nối với calibration |
| Calibration Bottleneck (ICML 2024, phía classification) | Cùng tinh thần "representation geometry → calibratability" nhưng đo qua training dynamics, không phải một lượt forward pass trên model đã đóng băng |
| NCTI/FaCe (ICCV 2023 / ICLR 2024) | Cùng công thức "NC-distance của frozen model → dự đoán property Y", nhưng Y = transferability, domain = classification |

---

## Rủi ro cần lường trước

- Giả định lý thuyết NRC (feature collapse vào subspace n chiều) chỉ được chứng minh chặt cho **Gaussian đơn**, chưa rõ áp dụng thế nào cho mixture density network — đúng như đã cảnh báo lượt trước, cần bắt đầu ở nhánh đơn giản nhất.
- 44/57 dataset của QRT có mức độ rời rạc (discreteness) đáng kể ở target — bản thân QRT đã ghi nhận NLL không phù hợp cho các dataset này; cần loại các dataset discreteness cao ra khỏi tập correlation chính (QRT đã tự làm việc này, có thể tái dùng đúng danh sách 13 dataset họ đã loại).
- Hiệu ứng của n (trục tương tự K) khó kiểm chứng mạnh vì phần lớn 57 dataset đều là n=1 — cần chủ động tìm thêm benchmark multivariate, không có sẵn số lượng lớn trong testbed hiện tại.

---

## Gợi ý phạm vi công bố

Giống khuyến nghị cho NCCS: bắt đầu ở dạng workshop paper (UDL, trustworthy-ML workshop tại ICML/NeurIPS/ICLR) vì phạm vi (một diagnostic + correction, kiểm chứng trên testbed có sẵn) phù hợp quy mô này hơn main track. Nếu Hướng 3 (trục n) cho kết quả mạnh và mới thêm được benchmark multivariate đáng kể, có thể nâng cấp thành submission đầy đủ, ghép chung với NCCS thành một bài báo duy nhất theo tinh thần "collapse-to-calibration" xuyên cả classification lẫn regression — đây sẽ là điểm nhấn hợp nhất rất mạnh cho UCF nói chung.

---

## Bước 0 — Chuẩn bị (trước khi chạy bất kỳ code nào)

**0.1. Lấy công thức NRC1/NRC2/NRC3 chính xác từ nguồn gốc.** Cần nói thẳng: mình chưa có công thức đầy đủ của NRC trong tay (khác với NC1-3 phía classification, mình đã nắm chắc công thức chuẩn từ Zhu et al. 2021). Phải đọc kỹ paper *"The Prevalence of Neural Collapse in Neural Multivariate Regression"* (NeurIPS 2024) và bản mở rộng *"Geometric Analysis of Neural Regression Collapse via Intrinsic Dimension"* trước khi viết bất kỳ dòng code nào — đây là việc bắt buộc làm tay, không thể bỏ qua hay đoán công thức.

**0.2. Setup môi trường.**
```
git clone https://github.com/Vekteur/quantile-recalibration-training
git clone https://github.com/Vekteur/probabilistic-calibration-study
```
Cài đúng dependency theo repo, kiểm tra checkpoint BASE có sẵn hay phải tự train lại (nếu phải train, không đáng ngại — model nhỏ, theo Table 2 trong paper QRT mỗi epoch chỉ mất từ 0.03 đến vài chục giây tùy dataset).

**0.3. Chọn tập pilot.** Không chạy hết 57 dataset ngay từ đầu. Chọn 8–10 dataset, ưu tiên:
- Loại bỏ 13 dataset có discreteness > 0.5 (danh sách có sẵn trong Table 3 phụ lục QRT)
- Chỉ dùng nhánh **BASE-MLP-1** (Gaussian đơn) — khớp đúng giả định lý thuyết NRC, tránh nhiễu do mixture
- Ưu tiên vài dataset nhỏ để chạy nhanh (CPU, Yacht, MPG, Boston, Concrete...) cho vòng thử đầu tiên trong ngày

---

## Bước 1 — Trích xuất feature & tính NRC (ngày 1–3)

1.1. Load checkpoint BASE của từng dataset pilot.
1.2. Forward pass qua **đúng tập calibration 15%** mà QRT đã chia sẵn (không tự chia lại, để số PCE đối chiếu được chính xác) — lấy feature tầng ngay trước lớp Linear cuối sinh ra (mean, std).
1.3. Tính NRC1/NRC2/NRC3 theo công thức đã xác nhận ở bước 0.1.
1.4. Ghi lại cho mỗi dataset: 1 con số NRC-distance tổng hợp + số PCE của BASE (đã có sẵn, lấy trực tiếp từ Appendix G/I của QRT, không cần tính lại).

---

## Bước 2 — GO/NO-GO thứ nhất (ngày 3–4)

2.1. Vẽ scatter: trục x = NRC-distance, trục y = PCE(BASE), mỗi điểm là 1 dataset pilot.
2.2. Tính **Spearman correlation** (không giả định tuyến tính) + p-value.
2.3. **Quy tắc quyết định cụ thể:**

| Kết quả | Hành động |
|---|---|
| \|ρ\| ≥ 0.5, p < 0.1 | **GO** — chuyển sang Bước 3 (full-scale) |
| 0.3 ≤ \|ρ\| < 0.5, hoặc p ≥ 0.1 nhưng xu hướng rõ trên đồ thị | **Mở rộng pilot lên ~20 dataset** trước khi quyết định dứt khoát — n=10 quá nhỏ để tin cậy |
| \|ρ\| < 0.3, không có xu hướng | **NO-GO** — dừng ở đây, không đầu tư thêm |

*(Lưu ý: p-value với n=10 có power rất thấp — đừng NO-GO quá sớm chỉ vì p ≥ 0.05, hãy nhìn hình dạng scatter trước.)*

---

## Bước 3 — Nếu GO: mở rộng full-scale (tuần 2)

3.1. Chạy hết 44 dataset còn lại (đã loại discreteness cao).
3.2. Lặp lại correlation ở quy mô đầy đủ, dùng đúng bộ công cụ thống kê QRT đã dùng để văn phong so sánh nhất quán: Friedman test → Wilcoxon signed-rank + Holm correction → critical difference diagram.
3.3. **Câu hỏi quan trọng hơn cả bước 2:** tính tương quan NRC-distance với PCE **sau khi** đã áp QR chuẩn (QRC), không chỉ PCE của BASE. Đây mới thực sự là phép đo "calibratability" (giống cách Calibration Bottleneck đo residual ECE sau Temperature Scaling) — nếu tương quan này vẫn giữ, tức NRC dự đoán được *cái gì QR không sửa nổi*, đó mới là phát hiện đáng giá nhất.
3.4. Lặp lại toàn bộ trên nhánh mixture-3-Gaussian và mixture-10-Gaussian — dùng checkpoint thật nếu upstream phát hành, nếu không phải train reproduction với seed/config/hash đầy đủ — để xem tương quan có sụp đổ khi giả định lý thuyết NRC bị vi phạm hay không. Đây là thông tin quan trọng để biết phạm vi áp dụng thật của phương pháp.

---

## Bước 4 — Nếu Bước 3 vẫn dương tính: xây correction (tuần 3–4)

4.1. Cài công thức calibration map điều chỉnh theo NRC-distance (post-hoc, không train lại).
4.2. So sánh trực tiếp với QRC/QRTC/QREGC bằng đúng letter-value plot + Cohen's d + critical difference diagram mà QRT đã dùng — tái dùng luôn code vẽ của họ.
4.3. Ablation: bỏ từng thành phần NRC1/NRC2/NRC3 riêng lẻ, xem thành phần nào đóng góp nhiều nhất — bắt buộc phải có để bài không bị hỏi "sao biết là do NRC chứ không phải trùng hợp".

---

## Nếu NO-GO ở Bước 2 (sau khi đã mở rộng pilot lên ~20)

Đừng bỏ trắng — hai lối ra:
- Viết thành **negative result có kiểm chứng** kèm phân tích: nếu cơ chế "NC-distance dự đoán calibratability" hoạt động tốt bên classification (NCCS) nhưng không giữ được ở regression, bản thân sự khác biệt đó là một phát hiện đáng công bố (ví dụ đưa ra giả thuyết vì sao — có thể do regression không có cấu trúc lớp rời rạc nên tín hiệu NC yếu hơn).
- Dồn lực sang hai hướng đã có sẵn: track H-NCCS (hierarchical) hoặc quay lại làm sâu track chính NCCS (classification) — không mất công vì hạ tầng đo NC đã dùng chung cho cả ba.

---

## Publication-grade protocol addendum

Phần này là ràng buộc bắt buộc sau khi audit implementation. Nó thay thế mọi cách hiểu mơ hồ trong thiết kế ban đầu.

### 1. Claim hierarchy

1. **Diagnostic claim:** NRC geometry measured on a frozen representation predicts held-out PCE across datasets/checkpoints.
2. **Residual-calibratability claim:** NRC geometry predicts PCE remaining after a standard QR baseline.
3. **Correction claim:** NRC-Cal improves PCE relative to BASE and a global-scale control without materially degrading NLL, CRPS, coverage, or sharpness.

Không được chuyển sang claim 2 hoặc 3 nếu claim trước đó không vượt qua gate đã đăng ký trước. Pilot synthetic chỉ dùng để kiểm tra pipeline; không được dùng làm evidence cho abstract hoặc conclusion.

### 2. No-leakage contract

- Chia train/validation/calibration/test trên dữ liệu thô trước khi fit preprocessing.
- Chỉ fit scaler, imputer, encoder trên train; validation/calibration/test chỉ được transform.
- Fit PCA basis, mean-head basis, NRC3, và mọi normalization statistic trên calibration split.
- Test-time sample distance phải là hàm của frozen feature và calibration-fitted geometry only; tuyệt đối không dùng test target và không fit lại PCA trên test.
- Hyperparameter của NRC-Cal phải được chọn bằng inner calibration holdout hoặc cross-fitting, sau đó refit trên toàn bộ calibration split.
- Test split chỉ được mở đúng một lần để tạo bảng cuối cùng.

### 3. Model and dataset protocol

- Primary cohort: BASE-MLP-1 Gaussian trên tập QRT không có target discreteness cao, dùng đúng split và seed upstream.
- Secondary cohorts: mixture-3, mixture-10, architecture/depth/width variants, và ít nhất ba checkpoint seeds.
- Multivariate cohort: tối thiểu ba dataset cho mỗi mức target dimension đủ khả thi; báo cáo cả trường hợp target covariance gần suy biến.
- Mỗi checkpoint phải có hash, upstream commit, preprocessing manifest, feature-layer identifier, mean-head weight, và immutable prediction cache.
- Việc clone repository không được xem là đã có checkpoint. Nếu upstream không phát hành checkpoint, quá trình train lại phải được khai báo như một reproduction stage riêng.

### 4. Required controls and baselines

- BASE, QR/QRC, QRTC, QREGC theo implementation upstream.
- Global variance scaling để kiểm tra NRC có thêm thông tin ngoài scale correction thông thường hay không.
- NRC1-only, NRC2-only, NRC1+NRC2; NRC3-only chỉ hợp lệ khi `n > 1`.
- Feature-free controls: predictive variance, residual norm trên calibration, dataset size, target skewness/kurtosis/discreteness, feature dimension, intrinsic dimension, and test RMSE.
- Khi compute budget cho phép: deep ensemble, MC dropout, Laplace hoặc một Bayesian baseline có implementation chuẩn. Các baseline này dùng để kiểm tra NRC-distance có chỉ đang proxy cho epistemic uncertainty hay không.

### 5. Statistical protocol

- Spearman là statistic chính cho diagnostic claim; Pearson và Kendall là sensitivity analyses.
- Báo cáo bootstrap confidence interval, permutation p-value, effect size, và probability that the paired effect is beneficial; không kết luận chỉ từ `p < 0.05`.
- So sánh nhiều methods bằng Friedman, paired Wilcoxon, và Holm correction; critical-difference diagram chỉ là visualization, không thay thế effect estimates.
- Với repeated seeds/model families, dùng hierarchical or mixed-effects model với random intercept cho dataset và checkpoint seed.
- Báo cáo power analysis trước full-scale. Nếu CI vẫn rộng và chứa effect nhỏ sau pilot, mở rộng số dataset/checkpoint thay vì diễn giải dấu của point estimate.

### 6. Robustness and sensitivity matrix

Full submission phải kiểm tra ít nhất các trục: seed, calibration size, feature dimension, intrinsic dimension, model depth/width, activation, optimizer, target dimension, heteroscedastic noise, label noise, missingness, covariate shift, target shift, và OOD severity. Báo cáo failure boundary cho zero/near-zero feature norms, rank-deficient mean head, singular target covariance, và extreme scale factors.

### 7. Publication gate

Main-conference claim chỉ hợp lệ khi:

- kết quả giữ trên real QRT datasets, không chỉ synthetic pilot;
- residual QR-PCE association có CI đủ hẹp và không bị giải thích hoàn toàn bởi dataset/model confounders;
- NRC-Cal vượt global-scale control với paired uncertainty rõ ràng;
- NLL/CRPS/sharpness không suy giảm đáng kể;
- code, manifest, hashes, seeds, environment, và raw result tables được export tự động.

Nếu một trong các điều kiện này thất bại, paper phải hạ claim thành diagnostic/negative-result study thay vì tiếp tục khẳng định correction tổng quát.

## Kaggle/Colab execution addendum

The runnable lab notebook uses Kaggle dataset `minhqunhc/aic-2026` as its primary data source and is designed for a Colab runtime with a Tesla T4. Before execution, the researcher must authenticate Kaggle in Colab with the `KAGGLE_API_TOKEN` Secret (or an uploaded `kaggle.json`) and confirm that dataset access/consent is granted. The notebook discovers tabular files, reports their schema, and requires an explicit `TARGET_COLUMN` override if automatic inference is not scientifically correct; it never substitutes synthetic data when Kaggle access fails. Missing feature values are imputed using medians fitted on the training split only.

This Kaggle run is an implementation and data-audit stage, not evidence for the full QRT-57 claim. Dataset-level Spearman correlation requires at least four independent files, groups, or checkpoints; a single eligible Kaggle table must be reported as `NOT ESTIMABLE`. Full publication claims still require the real frozen QRT checkpoint cohort, preregistered controls, and the no-leakage/statistical gates above.

### Kaggle dual-T4 execution note

Kaggle must assign the accelerator in the notebook Settings panel; `nvidia-smi` showing two T4 devices confirms assignment, but Python cannot enable a GPU that was not assigned. The notebook reports all visible devices and wraps the Gaussian MLP with `torch.nn.DataParallel` when two or more CUDA devices are visible. A one-GPU assignment remains valid for debugging, but final timing/results should record the actual GPU count in `outputs/environment.json`.
