import json
import numpy as np

def load_elem_db():
    return json.loads(open('../../elements.json').read())["elements"]

def load_elem():
    return [elem["symbol"] for elem in load_elem_db()]


def random_concentrations(n: int) -> np.ndarray:
    """
    Generate random concentrations that sum up to 100 using the Dirichlet distribution
    The values are between 5 and 60.
    """
    num_numbers = n  # You can change this to the desired number of proportions
    total = 100
    min_value = 5
    max_value = 60

    # Generate random proportions that sum up to 1 using the Dirichlet distribution
    proportions = np.random.dirichlet(np.ones(num_numbers), size=1)[0]

    # Scale the proportions to add up to 100
    scaled_numbers = [int(min_value + prop * (max_value - min_value)) for prop in proportions]

    # Calculate the difference between the total and the sum of the scaled numbers
    difference = total - sum(scaled_numbers)

    # Adjust the last number to ensure the sum is exactly 100
    scaled_numbers[-1] += difference

    return np.array(scaled_numbers)


def get_combinations(elements_sym, 
                     number_of_layers:int=2,
                     number_of_combinations:int=1,
                     lower:int=2,
                     upper:int=4,
                     ):
    '''
    Generates random combinations of 2 to 4 elements with the respective concentration.
    Bundles them into layers and returns a list of combinations.
    '''
    import random
    import numpy as np
    combination = []
    i = 0
    t = 0
    while i < number_of_combinations:
        t = 0
        layers = []
        while t < number_of_layers:
            n = random.randint(lower,upper)
            elem_perms = np.random.choice(elements_sym, size=n)
            conc_perms = random_concentrations(n)
            
            layers.append(list(zip(elem_perms, conc_perms)))
            t+=1
            
        combination.append(layers)
        i+=1
    yield combination

def kinetic_energy_list_to_binding_energy_al_kalpha(ke_list, round_dec: int = None):
    """
    Converts a list of kinetic energies (KE) to a list of binding energies (BE) in XPS
    using Al-K-alpha X-ray photons.

    Args:
        ke_list (list of float): List of kinetic energies in eV.

    Returns:
        list of float: List of binding energies in eV.
    """
    import math
    al_kalpha_energy = 1486.6  # Energy of Al-K-alpha X-ray photons in eV
    binding_energy_list = [round(al_kalpha_energy - ke, round_dec) for ke in ke_list]
    return np.array(binding_energy_list)

def binding_energy_list_to_kinetic_energy_al_kalpha(be_list, round_dec: int = None):
    """
    Converts a list of binding energies (BE) to a list of kinetic energies (KE) in XPS
    using Al-K-alpha X-ray photons.

    Args:
        be_list (list of float): List of binding energies in eV.

    Returns:
        list of float: List of kinetic energies in eV.
    """
    import math
    al_kalpha_energy = 1486.6  # Energy of Al-K-alpha X-ray photons in eV
    kinetic_energy_list = [round(al_kalpha_energy - be, round_dec) for be in be_list]
    return np.array(kinetic_energy_list)

def retreive_mlb_and_elements():
    from sklearn.preprocessing import MultiLabelBinarizer
    elements_db = json.loads(open('../../elements_sim.json').read())["elements"]
    elements = [elem["symbol"] for elem in elements_db]
    mlb = MultiLabelBinarizer()
    mlb.fit([elements])
    return mlb, elements

def one_hot_encode_concentrations(labels_and_values=[('Ar', 2), ('Si', 1), ('Mg', 5)]):
    import numpy as np
    import json
    mlb, _ = retreive_mlb_and_elements()
    # Step 2: Create a dictionary to map labels to indices
    label_to_index = {label: i for i, label in enumerate(mlb.classes_)}
    # Initialize the encoding vector with zeros
    encoding_vector = np.zeros(len(mlb.classes_))
    # Populate the encoding vector based on the labels and values
    for label, value in labels_and_values:
        index = label_to_index[label]
        encoding_vector[index] = value
    # Ensure that the values sum to 1
    encoding_vector = np.array(encoding_vector) / np.sum(encoding_vector)
    return encoding_vector

def pair_list_to_tuples(lst):
    return [(lst[i], lst[i + 1]) for i in range(0, len(lst), 2)]