from services import training_service

print("Testing all 11 experiment pipelines...")
for key, schema in training_service.EXPERIMENT_SCHEMAS.items():
    res = training_service.run_experiment_pipeline(key, test_size=0.2)
    cat = schema["category"]
    if cat == "Regression":
        m_str = f"R2={res['metrics'].get('r2_score')}, RMSE={res['metrics'].get('rmse')}"
    else:
        m_str = f"Acc={res['metrics'].get('accuracy')}%, F1={res['metrics'].get('f1_score')}%"
    print(f"[{key.upper()}] {schema['title']} -> {m_str} | Charts: {list(res['charts'].keys())}")

print("\nALL 11 PIPELINES COMPLETED SUCCESSFULLY!")
