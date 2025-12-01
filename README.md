<h1 align="center">
  <br>
  <a href="#"><img src="images/logo.svg" alt="Bionetta" width="250"></a>
  <br>
  Bionetta
  <br>
</h1>

<h4 align="center">Ultimate Client-Side ZKML Prover on top of <a href="https://www.tensorflow.org/" target="_blank">TensorFlow</a>.</h4>

<p align="center">
  <a href="#raising_hand-about">About</a> •
  <a href="#teacher-how-to-install">How To Install</a> •
  <a href="#test_tube-minimal-example">Example Usage</a> •
  <a href="LICENSE">License</a>
</p>

# :raising_hand: About

**Bionetta** :herb: is a client-side ZKML prover, currently built on top of TensorFlow. With **Bionetta** :herb:, you can prove a variety of statements of neural network execution:

- :white_check_mark: Maybe you want your users to prove that they can authenticate with their
_biometrics_ without revealing their _biometric data_?
- :white_check_mark: Maybe your neural network can perform the credit scoring and the user
wants to prove that they are creditworthy without revealing their _credit history_?
- :white_check_mark: Or maybe you want your users to submit the proof that the message or a photo
is not a spam nor a fake one (possibly with hiding the content of the message/photo)?

In such cases, **Bionetta** :herb: is your perfect choice! With it, you can generate proof of the neural network execution in tens of seconds, even on your mobile phone :iphone:, with easy on-chain verification (e.g., Ethereum). According to our benchmarks, the solution surpasses all current alternatives in the ZKML zoo.

> [!WARNING]
>
> Despite astounding performance, the library is **not** intended for 
> **private model weights** + **public input** proving. The library is designed only for **public model weights** + **private/public input** proving!

## :key: Key Features

- **Fast**: The fastest ZKML client-side prover on the market, with a proof generation time of less than 10 seconds even for a 1M parameter model.
- **Lightweight**: We are using
[_Groth16_](https://eprint.iacr.org/2016/260)-based provers (such as the
original Groth16 and customly written
[_UltraGroth_](https://hackmd.io/@Merlin404/Hy_O2Gi-h)), so the proof sizes are
constant and small (less than 1KB). Besides, the verification keys are typically
insignificant, so you can easily deploy them on-chain.
- **Easy to Use**: The library is built on top of TensorFlow, so you can use it with your existing models. You can also use the custom layers and callbacks to build your own architecture.

> [!NOTE]
>
> In the future releases, we are planning to add support for [PyTorch](https://pytorch.org/) as well.

## :racing_car: Key Benchmarks

Below, we specify the benchmarks for the following MNIST model with 2M
parameters and 6000 ReLU activations:

```python
def MNISTModel(inputs: tf.Tensor) -> tf.Tensor:
    """
    A simple network with ReLUs and Dense layers.
    """

    x = Flatten()(inputs)
    x = Dense(2000, activation="relu")(x)
    x = Dense(60, activation="relu")(x)
    x = Dense(4000, activation="relu")(x)
    outputs = Dense(10)(x)
    return outputs
```

To see the full comparison (including compilation time), you can read our
[Technical Report](docs/report.pdf). We use
[_UltraGroth_](https://hackmd.io/@Merlin404/Hy_O2Gi-h) as the base proving
system but note that Bionetta is compatible with any proving backend, supporting
R1CS.

| Metric                | Bionetta (UltraGroth)  | [ddkang/zkml](https://github.com/ddkang/zkml) | [EZKL](https://ezkl.xyz/) | [deep-prove](https://github.com/Lagrange-Labs/deep-prove) | [zkCNN](https://github.com/TAMUCrypto/zkCNN) |
|---------------------- |------------------------|----------------------|---------------------|------------|------------|
| Proof Size (KB)       | **0.88**               | 5.05                 | 127.0               | 8002.16    | 23.25      |
| Verification Key (MB) | **0.004 MB**           | 2.60                 | 4.10                | 0          | –          |
| Proving Key (GB)      | **0.20**               | 16.10                | 8.30                | 0          | –          |
| Proving Time (s)      | **3.05**               | 1100                 | 1310                | 3.66       | 3.45       |
| Verification Time (s) | **0.010**              | 0.012                | 5.40                | 0.522      | 1.00       |
| RAM Usage (GB)        | **0.27**               | 39.95                | 21.15               | 1.61       | 1.00       |


_For testing, we used an Intel Xeon E5-2665 (16 threads) CPU with 350 GB of RAM running Ubuntu 24.04.2 LTS (x86_64)._

## :teacher: How to Install?

**Recommended Python version**: 3.11. But other 
than 3.11, the library is not tested and might not work properly.


``` bash
python3.11 -m venv venv
source venv/bin/activate
bash install.sh

# Downlloads models that were used for benchmarks.
# NOTE: This may take some time.
bash load-models.sh

# Follow these instructions to launch the desired models
python3.11 -m examples.bench --help
```

> [!TIP]
> If, for some reason, your NPM requires `sudo` privileges, you can address [this Stack Overflow question](https://stackoverflow.com/questions/16151018/how-to-fix-npm-throwing-error-without-sudo) to fix the issue.

# :rocket: How it Works?

1. You create the _Keras_ model and either use it pre-trained or train it from scratch. The former is more desirable, as in such case you can design the
model architecture in a more flexible way, using _R1CS-friendly layers_.
The list of supported layers will be given below.
2. You wrap the model with the `BionettaModel` class, which will
    - Check the model architecture and print the summary of the constraints via the `constraints_summary` method.
    - Compile the model with the `compile_circuits` method, which will generate the proof circuits for the model.
    - Save the model weights and circuits to a specified directory.

That is pretty much it! Nonetheless, under the hood, the library does a lot of
heavy lifting, such as:

- **Model Architecture Check**: The library checks the model architecture and prints the summary of the constraints. It also checks if the model is R1CS-friendly and if not, specifies the layers with the most constraints.
- **Circom Code Generation**: The library generates the Circom code for the model, which is then compiled to the R1CS format. The Circom code is generated using the `circom` command-line tool, which is a part of the Circom library. We are also planning to add support for the [Noir](https://github.com/noir-lang/noir) language in the future.
- **Rust Witness Generator Generation**: The library generates the Rust code for the witness generator, which is then compiled to a binary. The Rust code is generated using the `rust` command-line tool, which is a part of the Rust library. Note
that our Rust code is specifically designed to be faster than the native
Circom solution.
- **Witness Generation and Proving**: The library generates the witness for the model using the Rust binary, which is then used to generate the proof. The proof is generated using the `snarkjs` command-line tool, which is a part of the SnarkJS library.

All the processes are visualized in the following diagram:

![Bionetta Architecture](./images/architecture.svg)

## :leafy_green: Supported Layers

| Layer Name | Notes | Implemented |
|------------|-------|-------------|
| `Dense` | Fully connected layer  | :white_check_mark: |
| `Conv2D`, no activation | Convolutional layer _with no activation function_ (using activation is very costly in R1CS)  | :white_check_mark: |
| `DepthwiseConv2D`, no activation | Depthwise convolutional layer _with no activation function_ (using activation is very costly in R1CS)  | :white_check_mark: |
| `Conv`s (with activations) | Convolutional/Depthwise layers _with no activation function_ (using activation is very costly in R1CS)  | :white_check_mark: |
| `EDLightConv2D` | Custom R1CS-friendly convolution  | :white_check_mark: |
| `SEHeavyBlock` | Modified [Squeeze-and-Excitation](https://arxiv.org/pdf/1709.01507) Block. Very R1CS-friendly.  | :white_check_mark: |
| Skip Connections | Residual Blocks. Might be unstable.  | :white_check_mark: |
| `Attention` | Attention Layer  | :x: |
| `Batch Normalization` | Batch normalization layer  | :white_check_mark: |
| `AvgPooling2D` | Average pooling layer  | :white_check_mark: |
| `SumPooling2D` | Sum pooling layer (average pooling without averaging)  | :white_check_mark: |
| `GlobalAvgPooling2D` | Global average pooling layer  | :white_check_mark: |
| `Flatten` | Flatten layer  | :white_check_mark: |
| `L2UnitNormalization` | L2 normalization layer (typically used in feature learning approaches)  | :white_check_mark: |
| `ReLU` | Rectified Linear Unit activation function  | :white_check_mark: |
| `LeakyReLU` | Leaky Rectified Linear Unit activation function. Supported only  for $\alpha = 2^{-s}$  | :white_check_mark: |
| `ReLU6` | Rectified Linear Unit activation function with upper bound  | :white_check_mark: |
| `HardSigmoid` | Hard Sigmoid activation function  | :white_check_mark: |
| `HardSwish` | Hard Swish activation function  | :white_check_mark: |

# :test_tube: Minimal Example

Here is the minimal example of how to use the library:

```python
import numpy as np
import tensorflow as tf
import tf_bionetta as tfb
from tf_bionetta.specs.backend_enums import ProvingBackend, WitnessGenerator
from tf_bionetta.specs.target import TargetPlatform


# 1. Load the MNIST dataset first
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = np.expand_dims(X_train, axis=-1)  # Add channel dimension
X_test = np.expand_dims(X_test, axis=-1)  # Add channel dimension
y_train = tf.keras.utils.to_categorical(y_train, 10) # One-hot encode the labels
y_test = tf.keras.utils.to_categorical(y_test, 10) # One-hot encode the labels

# 2. Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28,28,1)),
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(64),
    tfb.layers.ShiftReLU(5), # LeakyReLU with alpha=1/(2**5)=1/32
    tf.keras.layers.Dense(10)
], name='simple_mnist_model')

# 3. Wrap the model architecture with our custom BionettaModel class
proving_backend = ProvingBackend.GROTH16(15)

model = tfb.BionettaModel(model, verbose=2)
model.constraints_summary(proving_backend)
test_input = X_test[np.random.randint(len(X_test))]

# 4. Compile and train the model
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train,
    epochs=1,
    validation_data=(X_test, y_test),
    validation_split=0.2
)

# 5. Compile the model
model.compile_circuits(
    path='./examples/simple_mnist_circuits',
    test_input=test_input,
    save_weights=True,
    proving_backend=proving_backend,
    target_platform=TargetPlatform.DESKTOP,
    witness_backend=WitnessGenerator.CUSTOM,
    # Adds the XCFramework generation script, which can be used in iOS projects
    rust_witness_generator_options={
        RustGeneratorOptions.RUST_WITNESS_GENERATOR_XCFRAMEWORK_SCRIPTS_KEY: True
    }
)

# 6. Init your architecture from compiled dir
loaded_model = tfb.BionettaModel.load_from_compiled_folder('./examples/simple_mnist_circuits', verbose=2)
test_input = X_test[np.random.randint(len(X_test))]

# 7. Prove and verify your input. Simple mnist should be previousely compiled!!!
proof_dir = './examples/simple_mnist_circuits/proof'

proof = loaded_model.prove(
    input=test_input,
    target_dir=proof_dir,
)

print('The resultant proof:', proof)

assert loaded_model.verify(proof_dir=proof_dir), "model verification failed"
```

This launches the training and saves the model to the `./examples/simple_mnist_circuits` directory. To see more complex examples, 
see the [`examples`](./examples) directory.
