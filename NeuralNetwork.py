import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Activation functions and derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - x ** 2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_node_layers, output_nodes, activation_type):
        self.input_nodes = input_nodes
        self.hidden_node_layers = hidden_node_layers
        self.output_nodes = output_nodes
        self.activation_type = activation_type
        self.synapses = []
        self.thresholds = []
        self.node_values = []
        self.activation_functions = {
            "sigmoid": (sigmoid, sigmoid_derivative),
            "tanh": (tanh, tanh_derivative),
            "relu": (relu, relu_derivative),
        }
        self.initialize_synapses()

    def initialize_synapses(self):
        layers = [self.input_nodes] + self.hidden_node_layers + [self.output_nodes]
        for i in range(len(layers) - 1):
            limit = np.sqrt(2 / layers[i])
            self.synapses.append(np.random.uniform(-limit, limit, (layers[i], layers[i + 1])))
            self.thresholds.append(np.zeros(layers[i + 1]))
            self.node_values.append(np.zeros(layers[i]))
        self.node_values.append(np.zeros(layers[-1]))

    def propagate_forward(self, x):
        self.node_values[0] = x
        activation_func = self.activation_functions[self.activation_type][0]
        for i in range(len(self.synapses)):
            z = np.dot(self.node_values[i], self.synapses[i]) + self.thresholds[i]
            self.node_values[i + 1] = activation_func(z)
        return self.node_values[-1]

    def propagate_backward(self, x, y, learning_rate):
        m = x.shape[0]
        derivative_func = self.activation_functions[self.activation_type][1]
        delta = self.node_values[-1] - y
        for i in range(len(self.synapses) - 1, -1, -1):
            dW = np.dot(self.node_values[i].T, delta) / m
            db = np.sum(delta, axis=0) / m
            clip_value = 1.0
            dW = np.clip(dW, -clip_value, clip_value)
            db = np.clip(db, -clip_value, clip_value)
            self.synapses[i] -= learning_rate * dW
            self.thresholds[i] -= learning_rate * db
            if i > 0:
                delta = np.dot(delta, self.synapses[i].T) * derivative_func(self.node_values[i])

    def train(self, x, y, epochs, learning_rate, update_callback=None):
        self.error_history = []
        for epoch in range(epochs):
            output = self.propagate_forward(x)
            error = np.mean((output - y) ** 2)
            self.error_history.append(error)
            self.propagate_backward(x, y, learning_rate)
            if update_callback and (epoch % 10 == 0 or epoch == epochs - 1):
                update_callback()
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {error:.6f}")

    def visualize_error(self, canvas_frame):
        if hasattr(self, 'error_history') and self.error_history:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(range(1, len(self.error_history) + 1), self.error_history, label="Training Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training Loss Over Epoch")
            ax.legend()
            ax.grid(True)
            ax.set_ylim(bottom=0)  # Ensure the graph starts at 0

            for widget in canvas_frame.winfo_children():
                widget.destroy()

            chart_canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
            chart_canvas.draw()
            chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            print("No error data available to plot. Train the network first.")

class NeuralNetworkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Valued Neural Network")
        self.root.configure(bg="#333230")  # Set background color
        self.create_interface()

    def create_interface(self):
        frame = ttk.Frame(self.root, padding="20")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        input_frame = ttk.Frame(frame)
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        ttk.Label(input_frame, text="Configurations").grid(row=0, column=0, columnspan=2, pady=5)
        ttk.Label(input_frame, text="Input Nodes:").grid(row=1, column=0, sticky=tk.W)
        self.input_nodes = tk.IntVar(value=2)
        ttk.Entry(input_frame, textvariable=self.input_nodes, width=10).grid(row=1, column=1, sticky=tk.E)
        ttk.Label(input_frame, text="Hidden Layers:").grid(row=2, column=0, sticky=tk.W)
        self.hidden_layers = tk.StringVar(value="4,4")
        ttk.Entry(input_frame, textvariable=self.hidden_layers, width=10).grid(row=2, column=1, sticky=tk.E)
        ttk.Label(input_frame, text="Output Nodes:").grid(row=3, column=0, sticky=tk.W)
        self.output_nodes = tk.IntVar(value=1)
        ttk.Entry(input_frame, textvariable=self.output_nodes, width=10).grid(row=3, column=1, sticky=tk.E)

        # Added fields for Epochs and Learning Rate
        ttk.Label(input_frame, text="Epochs:").grid(row=4, column=0, sticky=tk.W)
        self.epochs = tk.IntVar(value=1000)
        ttk.Entry(input_frame, textvariable=self.epochs, width=10).grid(row=4, column=1, sticky=tk.E)
        ttk.Label(input_frame, text="Learning Rate:").grid(row=5, column=0, sticky=tk.W)
        self.learning_rate = tk.DoubleVar(value=0.01)
        ttk.Entry(input_frame, textvariable=self.learning_rate, width=10).grid(row=5, column=1, sticky=tk.E)

        activation_frame = ttk.Frame(frame)
        activation_frame.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)
        ttk.Label(activation_frame, text="Activation Function").grid(row=0, column=0, columnspan=2, pady=5)
        self.activation = tk.StringVar(value="relu")

        # Radio buttons for activation functions
        tk.Radiobutton(activation_frame, text="ReLU", variable=self.activation, value="relu", bg="#333230",
                       fg="white").grid(row=1, column=0, sticky=tk.W)
        tk.Radiobutton(activation_frame, text="Sigmoid", variable=self.activation, value="sigmoid", bg="#333230",
                       fg="white").grid(row=2, column=0, sticky=tk.W)
        tk.Radiobutton(activation_frame, text="Tanh", variable=self.activation, value="tanh", bg="#333230",
                       fg="white").grid(row=3, column=0, sticky=tk.W)

        action_frame = ttk.Frame(frame)
        action_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky=tk.W)
        ttk.Button(action_frame, text="Import Dataset", command=self.import_dataset).grid(row=0, column=0, padx=5,
                                                                                          pady=5)
        self.start_training_button = ttk.Button(action_frame, text="Begin training",
                                                command=self.start_training_with_dataset)
        self.start_training_button.grid(row=0, column=1, padx=5, pady=5)

        # Add reset button
        ttk.Button(action_frame, text="Reset", command=self.reset_network).grid(row=0, column=2, padx=5, pady=5)

        self.canvas = tk.Canvas(self.root, width=1000, height=500, bg="#333230")
        self.canvas.grid(row=2, column=0, columnspan=2, pady=10)

        self.graph_frame = tk.Frame(self.root, width=0, height=0, bg="#333230")
        self.graph_frame.grid(row=0, column=1, rowspan=3, sticky=tk.NW)

    def import_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                data = pd.read_csv(file_path)
                X = data.iloc[:, :-1].values
                y = data.iloc[:, -1].values.reshape(-1, 1)
                X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
                y = (y - y.min(axis=0)) / (y.max(axis=0) - y.min(axis=0))
                self.input_nodes.set(X.shape[1])
                self.output_nodes.set(y.shape[1])
                self.X, self.y = X, y
                self.generate_neuron()
                messagebox.showinfo(
                    "Dataset Imported",
                    f"Dataset imported successfully!\n\nInputs: {X.shape[1]}\nSamples: {X.shape[0]}"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import dataset:\n{str(e)}")

    def start_training_with_dataset(self):
        if not hasattr(self, 'neuron'):
            messagebox.showerror("Error", "Please import the dataset first!")
            return
        self.start_training_button.config(state='disabled')
        training_thread = threading.Thread(target=self.train_neuron)
        training_thread.start()

    def train_neuron(self):
        epochs = self.epochs.get()  # Use the value from the text field
        learning_rate = self.learning_rate.get()  # Use the value from the text field
        self.neuron.train(
            self.X,
            self.y,
            epochs=epochs,
            learning_rate=learning_rate,
            update_callback=self.update_visualization_thread_safe
        )
        self.root.after(0, self.on_training_complete)

    def generate_neuron(self):
        input_nodes = self.input_nodes.get()
        hidden_layers = list(map(int, self.hidden_layers.get().split(",")))
        output_nodes = self.output_nodes.get()
        activation = self.activation.get()
        self.neuron = NeuralNetwork(input_nodes, hidden_layers, output_nodes, activation)
        self.visualize_neuron()

    def visualize_neuron(self):
        self.canvas.delete("all")

        horizontal_padding = 0  # Padding on left and right
        vertical_padding = 0  # Padding on top and bottom
        global_y_offset = 30  # Shift everything downward by 30 pixels

        # Get the layers and dimensions
        layers = [self.neuron.input_nodes] + self.neuron.hidden_node_layers + [self.neuron.output_nodes]
        max_nodes = max(layers)
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        # Calculate gaps and positions
        layer_gap = width / len(layers)
        node_gap = height / max(2, min(max_nodes, 20))
        layer_positions = []

        for i, nodes in enumerate(layers):
            x = layer_gap * i + layer_gap / 2 + horizontal_padding
            layer_positions.append(
                [(x, vertical_padding + (height / 2) - (nodes / 2) * node_gap + j * node_gap + global_y_offset) for j in
                 range(nodes)])

        # Draw connections between nodes
        for i in range(len(layer_positions) - 1):
            for start_idx, node_start in enumerate(layer_positions[i]):
                for end_idx, node_end in enumerate(layer_positions[i + 1]):
                    self.canvas.create_line(
                        node_start[0], node_start[1],
                        node_end[0], node_end[1],
                        fill="gray", width=0.5
                    )

        # Draw the nodes
        for i, layer in enumerate(layer_positions):
            for j, (x, y) in enumerate(layer):
                self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill="white")

    def update_visualization_thread_safe(self):
        self.root.after(0, self.update_visualization_during_training)

    def update_visualization_during_training(self):
        self.canvas.delete("all")

        horizontal_padding = 0  # Padding on left and right
        vertical_padding = 0  # Padding on top and bottom
        global_y_offset = 30  # Shift everything downward by 30 pixels

        # Get the layers and dimensions
        layers = [self.neuron.input_nodes] + self.neuron.hidden_node_layers + [self.neuron.output_nodes]
        max_nodes = max(layers)
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        layer_gap = width / len(layers)
        node_gap = height / max(2, min(max_nodes, 20))
        layer_positions = []

        # Calculate node positions with vertical offset
        for i, nodes in enumerate(layers):
            x = layer_gap * i + layer_gap / 2 + horizontal_padding
            layer_positions.append(
                [(x, vertical_padding + (height / 2) - (nodes / 2) * node_gap + j * node_gap + global_y_offset) for j in
                 range(nodes)])

        # Draw connections between nodes with weights
        for i in range(len(layer_positions) - 1):
            max_weight = np.max(np.abs(self.neuron.synapses[i])) if np.max(np.abs(self.neuron.synapses[i])) != 0 else 1
            for start_idx, node_start in enumerate(layer_positions[i]):
                for end_idx, node_end in enumerate(layer_positions[i + 1]):
                    weight = self.neuron.synapses[i][start_idx, end_idx]
                    normalized_weight = abs(weight) / max_weight
                    line_width = max(1, int(normalized_weight * 5))
                    color_intensity = int(normalized_weight * 255)
                    if weight >= 0:
                        line_color = f"#{255 - color_intensity:02x}{color_intensity:02x}00"
                    else:
                        line_color = f"#{color_intensity:02x}00{255 - color_intensity:02x}"
                    self.canvas.create_line(
                        node_start[0], node_start[1],
                        node_end[0], node_end[1],
                        fill=line_color, width=line_width
                    )

        # Draw the nodes
        for i, layer in enumerate(layer_positions):
            for j, (x, y) in enumerate(layer):
                self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill="white")

    def reset_network(self):
        self.canvas.delete("all")
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        self.input_nodes.set(2)
        self.hidden_layers.set("4,4")
        self.output_nodes.set(1)
        self.epochs.set(1000)
        self.learning_rate.set(0.01)
        self.activation.set("relu")
        if hasattr(self, 'neuron'):
            del self.neuron
        if hasattr(self, 'X'):
            del self.X
        if hasattr(self, 'y'):
            del self.y

    def on_training_complete(self):
        messagebox.showinfo("Training Complete", "Training has completed successfully!")
        self.start_training_button.config(state='normal')
        self.neuron.visualize_error(self.graph_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkGUI(root)
    root.mainloop()
