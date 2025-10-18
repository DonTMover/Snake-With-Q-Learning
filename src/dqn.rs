#![cfg(feature = "dqn-gpu")]

use candle_core as candle;
use candle::Tensor;
use candle::Device;
use candle_nn as nn;
use candle_nn::{Module, VarBuilder, Optimizer};

const ACTIONS: usize = 3;

pub struct Replay {
    s: Vec<u32>,
    a: Vec<u8>,
    r: Vec<f32>,
    ns: Vec<u32>,
    done: Vec<u8>,
    cap: usize,
    idx: usize,
    full: bool,
}

impl Replay {
    pub fn new(cap: usize) -> Self {
        Self { s: Vec::with_capacity(cap), a: Vec::with_capacity(cap), r: Vec::with_capacity(cap), ns: Vec::with_capacity(cap), done: Vec::with_capacity(cap), cap, idx: 0, full: false }
    }
    pub fn push(&mut self, s: u32, a: u8, r: f32, ns: u32, done: bool) {
        if self.full {
            self.s[self.idx] = s;
            self.a[self.idx] = a;
            self.r[self.idx] = r;
            self.ns[self.idx] = ns;
            self.done[self.idx] = if done {1} else {0};
        } else {
            self.s.push(s); self.a.push(a); self.r.push(r); self.ns.push(ns); self.done.push(if done {1}else{0});
            if self.s.len() == self.cap { self.full = true; }
        }
        self.idx = (self.idx + 1) % self.cap;
    }
    pub fn len(&self) -> usize { if self.full { self.cap } else { self.s.len() } }
}

#[derive(Debug)]
pub struct DqnNet {
    emb: nn::Embedding,
    mlp1: nn::Linear,
    mlp2: nn::Linear,
    out: nn::Linear,
    device: Device,
}

impl DqnNet {
    pub fn new(vb: VarBuilder, device: &Device, state_vocab: usize, hidden: usize) -> candle::Result<Self> {
        // IMPORTANT: Scope variable names to avoid collisions across layers.
        let emb = nn::embedding(state_vocab, hidden, vb.clone().pp("emb"))?;
        let mlp1 = nn::linear(hidden, hidden, vb.clone().pp("mlp1"))?;
        let mlp2 = nn::linear(hidden, hidden, vb.clone().pp("mlp2"))?;
        let out = nn::linear(hidden, ACTIONS, vb.pp("out"))?;
        Ok(Self { emb, mlp1, mlp2, out, device: device.clone() })
    }
    pub fn q_values(&self, s_idx: &Tensor) -> candle::Result<Tensor> {
        // s_idx: [batch] (u32 mapped to index space)
        let x = self.emb.forward(s_idx)?;          // [batch, hidden]
        let x = x.relu()?;
        let x = self.mlp1.forward(&x)?.relu()?;
        let x = self.mlp2.forward(&x)?.relu()?;
        self.out.forward(&x)
    }
}

pub struct DqnAgent {
    pub net: DqnNet,
    pub opt: nn::AdamW,
    pub replay: Replay,
    pub gamma: f32,
    pub input_vocab: usize,
}

impl DqnAgent {
    pub fn new(input_vocab: usize, hidden: usize, device: &Device) -> candle::Result<Self> {
        let mut varmap = nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle::DType::F32, device);
        let net = DqnNet::new(vb, device, input_vocab, hidden)?;
        // Optimizer over all variables in the model
        let opt = nn::AdamW::new_lr(varmap.all_vars(), 1e-3)?;
        Ok(Self { net, opt, replay: Replay::new(20000), gamma: 0.99, input_vocab })
    }

    pub fn select_action(&self, state: u32) -> candle::Result<usize> {
    let s = Tensor::new(&[state % self.input_vocab as u32], &self.net.device)?; // [1]
        let q = self.net.q_values(&s)?; // [1, 3]
        let idxs = q.argmax(1)?; // indices along dim=1, shape [1]
        let v = idxs.to_vec1::<i64>()?;
        Ok(v[0] as usize)
    }

    pub fn push_transition(&mut self, s: u32, a: usize, r: f32, ns: u32, done: bool) {
        self.replay.push(s, a as u8, r, ns, done);
    }

    pub fn train_step(&mut self, batch: usize) -> candle::Result<()> {
        let n = self.replay.len();
        if n < batch { return Ok(()); }
        // Sample first `batch` items (simple; can be improved with RNG)
        let s: Vec<u32> = self.replay.s.iter().cloned().take(batch).collect();
        let a: Vec<i64> = self.replay.a.iter().map(|&x| x as i64).take(batch).collect();
        let r: Vec<f32> = self.replay.r.iter().cloned().take(batch).collect();
        let ns: Vec<u32> = self.replay.ns.iter().cloned().take(batch).collect();
        let done: Vec<f32> = self.replay.done.iter().map(|&d| d as f32).take(batch).collect();

        let dev = &self.net.device;
        let s_t = Tensor::new(&s[..], dev)?;               // [B]
        let a_t = Tensor::new(&a[..], dev)?;               // [B]
        let r_t = Tensor::new(&r[..], dev)?;               // [B]
        let ns_t = Tensor::new(&ns[..], dev)?;             // [B]
        let done_t = Tensor::new(&done[..], dev)?;         // [B]
        let q = self.net.q_values(&s_t)?;                  // [B, 3]
        let q_a = q.gather(&a_t.unsqueeze(1)?, 1)?         // [B,1]
            .squeeze(1)?;                                  // [B]
    let nq = self.net.q_values(&ns_t)?;                // [B,3]
    let max_nq = nq.max(1)?.squeeze(1)?;               // [B]
    // Build tensors for scalar/broadcast ops
    let bsz = s.len();
    let ones = Tensor::ones(&[bsz], candle::DType::F32, dev)?; // [B]
    let not_done = (&ones - &done_t)?;                        // [B]
    let gamma_t = Tensor::new(self.gamma, dev)?;              // scalar
    let gamma_nq = (&max_nq * &gamma_t)?;                     // [B]
    let target = (&r_t + (&not_done * &gamma_nq)?)?;          // [B]
        let loss = (q_a - target)?.sqr()?.mean(0)?;        // MSE

        self.opt.backward_step(&loss)?;
        Ok(())
    }
}

pub fn preferred_device() -> Device {
    // Try CUDA if feature enabled, else CPU
    #[cfg(feature = "dqn-gpu-cuda")]
    if let Ok(dev) = Device::new_cuda(0) { return dev; }
    Device::Cpu
}
