# MLOps capstone flow skeleton starter: manual monitoring loop for green taxi tip prediction.
# Runs: load new batch -> integrity gate -> evaluate champion -> (optional) retrain -> register -> (optional) promote.

from metaflow import FlowSpec, Parameter, step


class MLFlowCapstoneFlow(FlowSpec):
    reference_path = Parameter("reference-path")
    batch_path = Parameter("batch-path")
    model_name = Parameter("model-name", default="green_taxi_tip_model")

    @step
    def start(self):
        init_mlflow(self.model_name)
        self.next(self.load_data)

    @step
    def load_data(self):
        self.ref, self.batch = load_reference(self.reference_path), load_batch(self.batch_path)
        self.next(self.integrity_gate)

    @step
    def integrity_gate(self):
        ok, report = run_integrity_checks(self.ref, self.batch)  # hard + NannyML
        self.next(self.load_champion if ok else self.end)

    @step
    def load_champion(self):
        # TODO: Add relevant steps and flow logic.
        pass


if __name__ == "__main__":
    MLFlowCapstoneFlow()
