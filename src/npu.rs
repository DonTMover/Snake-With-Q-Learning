#![cfg(all(target_os = "windows", feature = "npu-directml"))]

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use ort::{environment::Environment, session::SessionBuilder, tensor::OrtOwnedTensor, LoggingLevel, GraphOptimizationLevel};

pub struct NpuPolicy {
    env: Environment,
    session: ort::Session,
    input_name: String,
    output_name: String,
    input_vocab: usize,
    actions: usize,
}

impl NpuPolicy {
    pub fn load(model_path: &str, input_vocab: usize, actions: usize) -> Result<Self> {
        let env = Environment::builder()
            .with_name("snake-npu")
            .with_log_level(LoggingLevel::Warning)
            .build()?;

        // Prefer DirectML provider on Windows NPU
        let session = SessionBuilder::new(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_directml()? // Enable DML Execution Provider
            .commit(model_path)?;

        let inputs = session.inputs();
        let outputs = session.outputs();
        if inputs.is_empty() || outputs.is_empty() {
            return Err(anyhow!("ONNX model must have at least 1 input and 1 output"));
        }
        let input_name = inputs[0].name.clone();
        let output_name = outputs[0].name.clone();

        Ok(Self {
            env,
            session,
            input_name,
            output_name,
            input_vocab,
            actions,
        })
    }

    pub fn select_action(&self, state: u32) -> Result<usize> {
        // We treat the state as a categorical index (embedding should be in the model)
        let idx = (state as usize) % self.input_vocab;
        let input: Array2<i64> = Array2::from_shape_vec((1, 1), vec![idx as i64])?; // shape [1,1] index
        let outputs: Vec<OrtOwnedTensor<f32, _>> = self
            .session
            .run(ort::inputs!{ self.input_name.clone() => input }?)?;
        if outputs.is_empty() {
            return Err(anyhow!("ONNX inference returned no outputs"));
        }
        let logits = outputs[0].view();
        // Expect [1, actions]
        let row = logits.index_axis(ndarray::Axis(0), 0);
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, v) in row.iter().enumerate() {
            if *v > best_v {
                best_v = *v;
                best = i;
            }
        }
        Ok(best.min(self.actions.saturating_sub(1)))
    }
}
