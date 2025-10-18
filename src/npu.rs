#![cfg(all(target_os = "windows", feature = "npu-directml"))]

use anyhow::{anyhow, Result};
use ndarray::{Array2};
use ort::execution_providers::DirectMLExecutionProvider;
use ort::logging::LogLevel;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

pub struct NpuPolicy {
    session: Session,
    input_name: String,
    output_name: String,
    input_vocab: usize,
    actions: usize,
}

impl NpuPolicy {
    pub fn load(model_path: &str, input_vocab: usize, actions: usize) -> Result<Self> {
        // Build a session preferring DirectML on Windows
        let session = Session::builder()?
            .with_log_level(LogLevel::Warning)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([
                DirectMLExecutionProvider::default().build(),
            ])?
            .commit_from_file(model_path)?;

    let inputs = &session.inputs;
    let outputs = &session.outputs;
        if inputs.is_empty() || outputs.is_empty() {
            return Err(anyhow!("ONNX model must have at least 1 input and 1 output"));
        }
        let input_name = inputs[0].name.clone();
        let output_name = outputs[0].name.clone();

        Ok(Self {
            session,
            input_name,
            output_name,
            input_vocab,
            actions,
        })
    }

    pub fn select_action(&mut self, state: u32) -> Result<usize> {
        // We treat the state as a categorical index (embedding should be in the model)
        let idx = (state as usize) % self.input_vocab;
        // shape [1, 1] index tensor (int64)
        let input: Array2<i64> = Array2::from_shape_vec((1, 1), vec![idx as i64])?;
        let input_tensor = TensorRef::from_array_view(&input)?;

    let outputs = self.session.run(ort::inputs! { self.input_name.as_str() => input_tensor })?;

        // Extract output tensor as ndarray array (f32)
        let logits = outputs[self.output_name.as_str()].try_extract_array::<f32>()?;
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
