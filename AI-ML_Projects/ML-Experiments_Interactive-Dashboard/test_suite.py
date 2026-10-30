import io
import sys
import app

client = app.app.test_client()

print("==================================================")
print("ADVANCED ML EXPERIMENTS LAB - EXTENDED TEST SUITE")
print("==================================================")

# --- 1. Testing Core Navigation GET Routes ---
print("\n--- 1. Testing Core Navigation GET Routes ---")
routes = [
    ("/", "Overview"),
    ("/experiments", "Experiments Library"),
    ("/experiments?category=Regression", "Regression Category Filter"),
    ("/experiments?category=Classification", "Classification Category Filter"),
    ("/datasets", "Dataset Explorer"),
    ("/datasets?file=Placement.csv", "Dataset Explorer (Placement.csv)"),
    ("/training", "Model Training Studio"),
    ("/evaluation", "Model Evaluation Default"),
    ("/comparison", "Model Comparison Studio"),
    ("/explainability", "Explainable AI (XAI) Studio"),
    ("/reports", "Academic Lab Reports"),
    ("/history", "Training History & Audit Log"),
    ("/learning", "ML Learning Mode & Viva Prep"),
    ("/about", "About Project & Settings")
]

for url, label in routes:
    res = client.get(url)
    assert res.status_code == 200, f"Route {url} failed with status {res.status_code}"
    html = res.get_data(as_text=True)
    assert "Prathamesh Balaji Patil" in html, f"Student name missing in {url}"
    print(f"[OK] 200 OK: {label} ({url})")

# --- 2. Testing All 11 Individual Prediction Form Routes (GET) ---
print("\n--- 2. Testing All 11 Individual Prediction Form Routes (GET) ---")
for key, spec in app.ms.ALGORITHMS.items():
    res = client.get(f"/algorithm/{key}")
    assert res.status_code == 200, f"Algorithm {key} GET failed"
    html = res.get_data(as_text=True)
    assert spec["title"] in html, f"Title {spec['title']} missing in /algorithm/{key}"
    print(f"[OK] 200 OK: Exp {spec['num']} ({key}): {spec['title']}")

# --- 3. Testing Real Model Predictions on All 11 Algorithms (POST) ---
print("\n--- 3. Testing Real Model Predictions on All 11 Algorithms (POST) ---")
for key, spec in app.ms.ALGORITHMS.items():
    sample = spec.get("sample_data", {})
    res = client.post(f"/algorithm/{key}", data=sample)
    assert res.status_code == 200, f"Algorithm {key} prediction POST failed"
    html = res.get_data(as_text=True)
    assert "Prediction Complete" in html, f"Prediction Complete indicator missing in {key}"
    print(f"[OK] Inference Verified: Exp {spec['num']} ({key}) -> Success!")

# --- 4. Testing Input Validation & Error Handling (Invalid & Empty Inputs) ---
print("\n--- 4. Testing Input Validation & Error Handling ---")
for key, spec in app.ms.ALGORITHMS.items():
    # Empty POST
    res_empty = client.post(f"/algorithm/{key}", data={})
    assert res_empty.status_code == 200, f"Empty post failed on {key}"
    html_empty = res_empty.get_data(as_text=True)
    assert "Please provide a value" in html_empty or "Please enter message" in html_empty or "Invalid numeric input" in html_empty
    
    # Non-numeric POST (except message which accepts text)
    if key != "message":
        bad_sample = {k: "not_a_number" for k in spec.get("sample_data", {})}
        res_bad = client.post(f"/algorithm/{key}", data=bad_sample)
        assert res_bad.status_code == 200
        html_bad = res_bad.get_data(as_text=True)
        assert "Invalid numeric input" in html_bad or "Please provide a value" in html_bad

    print(f"[OK] Input Validation Verified: Exp {spec['num']} ({key})")

# --- 5. Testing Results & Visualizations Routes for All 11 Algorithms ---
print("\n--- 5. Testing Results & Visualizations Routes for All 11 Algorithms ---")
for key, spec in app.ms.ALGORITHMS.items():
    res = client.get(f"/results/{key}")
    assert res.status_code == 200, f"Results route for {key} failed"
    html = res.get_data(as_text=True)
    assert spec["title"] in html, f"Title missing in results for {key}"
    print(f"[OK] Results & Visualizations: Exp {spec['num']} ({key}) OK!")

# --- 6. Testing Explainable AI (XAI) Routes for All Modules ---
print("\n--- 6. Testing Explainable AI (XAI) Routes for All Modules ---")
for key in ["exploration"] + list(app.ms.ALGORITHMS.keys()):
    res = client.get(f"/explainability/{key}")
    assert res.status_code == 200, f"Explainability route for {key} failed"
    html = res.get_data(as_text=True)
    assert "Global Feature Importance" in html
    print(f"[OK] XAI Lab Verified: ({key}) OK!")

# --- 7. Testing Academic Lab Reports Routes for All 11 Algorithms ---
print("\n--- 7. Testing Academic Lab Reports Routes for All 11 Algorithms ---")
for key, spec in app.ms.ALGORITHMS.items():
    res = client.get(f"/reports/{key}")
    assert res.status_code == 200, f"Reports route for {key} failed"
    html = res.get_data(as_text=True)
    assert spec["title"] in html, f"Title missing in reports for {key}"
    print(f"[OK] Report Preview Verified: Exp {spec['num']} ({key}) OK!")

# --- 8. Testing Learning Mode Routes for All 12 Modules ---
print("\n--- 8. Testing Learning Mode Routes for All 12 Modules ---")
for key, item in app.ls.LEARNING_DATA.items():
    res = client.get(f"/learning/{key}")
    assert res.status_code == 200, f"Learning route for {key} failed"
    html = res.get_data(as_text=True)
    assert "Viva Voce" in html
    print(f"[OK] Learning Mode Verified: Exp {item.get('exp_num', key)} ({key}) OK!")

# --- 9. Testing REST API Endpoints ---
print("\n--- 9. Testing REST API Endpoints ---")
# 1. Dataset Preview API
prev_res = client.get("/api/dataset-preview/heart")
assert prev_res.status_code == 200
prev_data = prev_res.get_json()
assert prev_data["rows"] > 0
print(f"[OK] API /api/dataset-preview/heart: Rows = {prev_data['rows']}, Cols = {prev_data['columns']}")

# 2. Dataset Schema API
schema_res = client.get("/api/dataset-schema/Placement.csv")
assert schema_res.status_code == 200
schema_data = schema_res.get_json()
assert schema_data["success"] is True
print(f"[OK] API /api/dataset-schema/Placement.csv: Columns = {schema_data['schema']['columns']}")

# 3. Model Training API (Classification)
train_res = client.post("/api/train", json={"key": "heart", "test_size": 0.2, "random_state": 42})
assert train_res.status_code == 200
train_data = train_res.get_json()
assert train_data["success"] is True
assert "accuracy" in train_data["metrics"]
print(f"[OK] API /api/train (heart): Accuracy = {train_data['metrics']['accuracy']}%")

# 4. Model Training API (Regression)
train_reg_res = client.post("/api/train", json={"key": "slr", "test_size": 0.2, "random_state": 42})
assert train_reg_res.status_code == 200
train_reg_data = train_reg_res.get_json()
assert train_reg_data["success"] is True
assert "r2_score" in train_reg_data["metrics"]
print(f"[OK] API /api/train (slr): R2 Score = {train_reg_data['metrics']['r2_score']}")

# 5. Model Prediction API
pred_res = client.post("/api/predict/slr", json={"cgpa": "8.5"})
assert pred_res.status_code == 200
pred_data = pred_res.get_json()
assert pred_data["success"] is True
print(f"[OK] API /api/predict/slr: Output = {pred_data['result']['prediction']}")

# 6. Explain API
explain_res = client.post("/api/explain/heart", json={"age": "55", "sex": "1", "cp": "2", "trestbps": "130"})
assert explain_res.status_code == 200
explain_data = explain_res.get_json()
assert explain_data["success"] is True
print(f"[OK] API /api/explain/heart: Details count = {len(explain_data['explanation']['details'])}")

# 7. Model Comparison API
comp_res = client.post("/api/compare", json={
    "dataset": "data_banknote_authentication.csv",
    "target": "class",
    "algorithms": ["logistic_regression", "linear_svm", "decision_tree"],
    "test_size": 0.2,
    "random_state": 42
})
assert comp_res.status_code == 200
comp_data = comp_res.get_json()
assert comp_data["success"] is True
assert len(comp_data["data"]["results"]) == 3
print(f"[OK] API /api/compare: Evaluated 3 models, Best = {comp_data['data']['results'][0]['name']}")

# 8. PDF Lab Report Generation API (All 11 Algorithms)
print("\n--- 10. Testing PDF Report Generation for All 11 Experiments ---")
for k in app.ms.ALGORITHMS.keys():
    pdf_res = client.get(f"/api/export-pdf/{k}")
    assert pdf_res.status_code == 200
    assert pdf_res.mimetype == "application/pdf"
    assert pdf_res.data.startswith(b"%PDF"), f"Invalid PDF header for {k}"
    print(f"[OK] PDF Lab Report Verified: {k} ({len(pdf_res.data)} bytes)")

# 9. Export JSON Report API
export_json = client.get("/api/export-report/slr?format=json")
assert export_json.status_code == 200
assert export_json.mimetype == "application/json"
print("[OK] API /api/export-report?format=json: Success")

# 10. Export CSV Report API
export_csv = client.get("/api/export-report/slr?format=csv")
assert export_csv.status_code == 200
assert export_csv.mimetype == "text/csv"
print("[OK] API /api/export-report?format=csv: Success")

# 11. Profile Save API
prof_res = client.post("/api/save-profile", json={"name": "Prathamesh Balaji Patil", "roll_no": "A11"})
assert prof_res.status_code == 200
assert prof_res.get_json()["success"] is True
print("[OK] API /api/save-profile: Success")

# 12. Error Handlers
res_404 = client.get("/nonexistent-page-url")
assert res_404.status_code == 404
print("[OK] Custom 404 HTML Error Handler: Success")

res_api_404 = client.get("/api/nonexistent-api-endpoint")
assert res_api_404.status_code == 404
assert res_api_404.is_json
print("[OK] Custom 404 API JSON Error Handler: Success")

print("\n==================================================")
print("ALL VERIFICATION SUITE CHECKS PASSED (100% SUCCESS)")
print("==================================================")
