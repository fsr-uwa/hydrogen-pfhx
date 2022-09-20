# main file of hydrogen-pfhx

import yaml
import numpy as np
from scipy.integrate import solve_bvp
import fluids, catalysts, hexs, bvp_model, helium_neon
from outputs import post_process
from utils import tpd_to_kps


def model(configuration_file):
    # Step 1. read config file
    if not configuration_file:
        # use default config file if not specified.
        configuration_file = 'hydrogen_pfhx/configs/default_configuration.yaml'
        
    with open(configuration_file, "r") as stream:
        try:
            configuration = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Step 2. Create a reactant, coolant, reactor & catalyst
    # reactant
    reactant_mass_flowrate_kps = tpd_to_kps(
        configuration['reactant']['mass_flow_rate'])
    reactant = fluids.Hydrogen(reactant_mass_flowrate_kps)

    # coolant
    coolant_name = configuration['coolant']['fluid']
    reactor_mass_flowrate_kps = tpd_to_kps(
        configuration['coolant']['mass_flow_rate'])
    if coolant_name == 'Hydrogen':
        xp_cool_inlet = configuration['coolant']['x_para']
        coolant = fluids.Hydrogen(reactor_mass_flowrate_kps)
        coolant.update_composition(xp_cool_inlet)
    elif coolant_name == 'HeliumNeon':
        helium_neon.setup_CoolProp(['Helium', 'Neon', 'Argon'])
        helium_fraction = configuration['coolant']['helium_fraction']
        coolant = fluids.HeliumNeon(reactor_mass_flowrate_kps, helium_fraction)
    else:
        coolant = fluids.FluidStream(coolant_name, reactor_mass_flowrate_kps)

    # reactor
    reactor = hexs.PlateFinHex(configuration['reactor'])

    # catalyst
    catalyst = catalysts.Catalyst(configuration['catalyst'])

    # Step 3. Setup the boundary value problem - initialise a solution
    boundary_properties = np.array(
        configuration['reactant']['x_para'],
        configuration['reactant']['pressure'],
        configuration['reactant']['temperature'],
        configuration['coolant']['pressure'],
        configuration['coolant']['temperature'])

    (x_mesh, sol_init) = bvp_model.initialise_solution(reactant, coolant,
                                                       reactor, catalyst, boundary_properties, configuration['simulation'])

    # Step 4. Run the BVP solver
    additional_parameters = (reactor, catalyst, reactant,
                             coolant, boundary_properties)
    solution = solve_bvp(lambda x, y: bvp_model.bvp_function(x, y, additional_parameters=additional_parameters),
                         lambda xb, yb: bvp_model.counter_current_boundary_condition(
                             xb, yb, boundary_properties),
                         x_mesh, sol_init, tol=configuration['simulation']['tolerance'], max_nodes=1000, verbose=2)

    # Step 5. Post-process & plot results.
    results = post_process(solution)

    return results



if __name__ == '__main__':
    # setup configuration file argument
    import argparse
    parser = argparse.ArgumentParser(description='Configuration file setup.')
    parser.add_argument('--configuration_file', metavar='path', required=False,
                        help='path to config file')
    args = parser.parse_args()

    # run model
    results = model(configuration_file=args.configuration_file)
