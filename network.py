from graphviz import Digraph

def create_complex_neural_network():
    dot = Digraph()

    # Input Layer
    input_neurons = ['I1', 'I2', 'I3']
    for i in input_neurons:
        dot.node(i, f'Input {i[-1]}')

    # Hidden Layer 1
    hidden1_neurons = ['H1_1', 'H1_2']
    for h1 in hidden1_neurons:
        dot.node(h1, f'Hidden 1 {h1[-1]}')

    # Hidden Layer 2
    hidden2_neurons = ['H2_1', 'H2_2']
    for h2 in hidden2_neurons:
        dot.node(h2, f'Hidden 2 {h2[-1]}')

    # Output Layer
    output_neurons = ['O1']
    for o in output_neurons:
        dot.node(o, f'Output {o[-1]}')

    # Edges from Input Layer to Hidden Layer 1
    for i in input_neurons:
        for h1 in hidden1_neurons:
            dot.edge(i, h1)

    # Edges from Hidden Layer 1 to Hidden Layer 2
    for h1 in hidden1_neurons:
        for h2 in hidden2_neurons:
            dot.edge(h1, h2)

    # Edges from Hidden Layer 2 to Output Layer
    for h2 in hidden2_neurons:
        for o in output_neurons:
            dot.edge(h2, o)

    return dot

network = create_complex_neural_network()
network.render('complex_neural_network', format='png', view=True)
