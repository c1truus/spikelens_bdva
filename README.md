# SpikeLens BDVA

SpikeLens is an interactive visualization project for exploring how a spiking neural network transforms latency-coded MNIST input into class decisions over time.

The deployed webapp is a static Vite/Svelte site. It does not run PyTorch or snnTorch in the browser. Instead, inference traces are precomputed offline and exported as JSON files under:

```text
spikelens_webapp/public/data/
```
