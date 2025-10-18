#![cfg(feature = "gpu-nn")]

// The experimental gpu-nn scaffolding has been disabled and its dependencies were removed.
// If you see this error, please remove the `gpu-nn` feature from your build.
compile_error!("The 'gpu-nn' scaffolding is currently disabled. Do not enable the 'gpu-nn' feature.");

// Disabled code below remains for reference; it requires the Burn crates.
// use burn::backend::Autodiff;
// use burn::backend::wgpu::{AutoGraphicsApi, Wgpu, WgpuDevice};
// use burn::module::Module;
// use burn::nn::{Linear, LinearConfig, Relu};
// use burn::tensor::{Tensor, activation::softmax};

type B = Autodiff<Wgpu<AutoGraphicsApi, f32, i32>>;

#[derive(Module, Debug)]
pub struct PolicyNet {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc_out: Linear<B>,
}

impl PolicyNet {
    pub fn new(input: usize, hidden: usize, output: usize) -> Self {
        let cfg1 = LinearConfig::new(input, hidden);
        let cfg2 = LinearConfig::new(hidden, hidden);
        let cfg3 = LinearConfig::new(hidden, output);
        Self {
            fc1: cfg1.init(),
            fc2: cfg2.init(),
            fc_out: cfg3.init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        let x = Relu::new().forward(x);
        let x = self.fc2.forward(x);
        let x = Relu::new().forward(x);
        let x = self.fc_out.forward(x);
        softmax(x, 1)
    }
}

pub struct GpuTrainer {
    pub device: WgpuDevice,
    pub net: PolicyNet,
}

impl GpuTrainer {
    pub fn new(input: usize, hidden: usize, output: usize) -> Self {
        let device = WgpuDevice::BestAvailable;
        let net = PolicyNet::new(input, hidden, output);
        Self { device, net }
    }

    // Encode a batch of states into a tensor [batch, input]
    pub fn encode_states(&self, batch_states: &[u32], input: usize) -> Tensor<B, 2> {
        // Placeholder: simple one-hot by (state % input)
        let batch = batch_states.len();
        let mut data = vec![0.0f32; batch * input];
        for (i, s) in batch_states.iter().enumerate() {
            let idx = (*s as usize) % input;
            data[i * input + idx] = 1.0;
        }
        Tensor::<B, 2>::from_floats(data, [batch, input])
    }

    // Inference: returns action probabilities [batch, actions]
    pub fn infer(&self, batch_states: &[u32], input: usize, actions: usize) -> Tensor<B, 2> {
        let x = self.encode_states(batch_states, input);
        let probs = self.net.forward(x);
        assert_eq!(probs.dims(), [batch_states.len(), actions]);
        probs
    }

    // Convenience: run inference and return a flat Vec<f32> of size batch*actions (row-major)
    pub fn infer_to_vec(&self, batch_states: &[u32], input: usize, actions: usize) -> Vec<f32> {
        let probs = self.infer(batch_states, input, actions);
        let data = probs.into_data();
        let vals: Vec<f32> = data.convert::<f32>().value;
        // Expect len == batch*actions
        debug_assert_eq!(vals.len(), batch_states.len() * actions);
        vals
    }
}
