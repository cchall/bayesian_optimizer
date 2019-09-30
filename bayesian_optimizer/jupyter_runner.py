import os
import numpy as np
import uuid, time, yaml
import subprocess
import h5py as h5
from .acquisition import MyBounds


class Optimizer:
    def __init__(self, optimizer, dim, bounds, n_proc, suffix_path):
        self.optimizer = optimizer(dim)
        self.n_proc = n_proc  # Processors per objective evaluation
        self.server_log = {i: None for i in range(n_proc)}
        self.current_simulations = {}
        self.X = None
        self.Y = None
        self.path = os.path.join(os.getcwd(), suffix_path)
        self.run_string = 'python2 client.py simulation run_warp.py --name {name} --cores {cores} --arguments {args} --path {path}'
        self.bounds = MyBounds(bounds[0], bounds[1])

    def load_data(self, X, y=None):
        self.X = X
        if not np.any(y):
            self.Y = np.empty_like(X)
        else:
            self.Y = y

    def dispatch(self, identifiers):
        """
        Send out simulation jobs to the server
        Runs out of the directory given by self.path so the parameter file shouldn't need a directory path
        if it was put where it belongs.
        """

        for idents in identifiers:
            parameter_file = idents + '.yaml'
            # Dispatch job to server
            subprocess.Popen(
                self.run_string.format(name=idents, cores=self.n_proc, args=parameter_file, path=self.path), shell=True)
            # Log the job as running
            self.current_simulations[idents] = None
            time.sleep(0.5)  # Not sure if server can be overloaded, won't take a chance for now

    def optimize(self, samples, batches, cycles=1, T=0.15, hierarchical=False):
        """
        Run N cycles of M batches, each composed of L samples. Each batch is run as a full unit and the number of batches determines
        the spread in settings for UCB exploit/explore parameter. We will sacrifice full asynchronicity to wait until a batch is ready
        to go to better diversify trial observations.

        Open question is how to select UCB EE parameter after the first set is queued. Going to pick a batch at random for now.
        """
        # Initial model training
        model = self.optimizer.train(self.X, self.Y)
        # Initial point selection
        observation_points, _ = self.optimizer.choose(self.bounds, samples=samples, batches=batches, T=T,
                                                      hierarchy=hierarchical)
        observation_points = np.array(observation_points).reshape(batches, samples, -1)
        # Create files then send jobs to server
        identifier_pool = []
        for batch_set in observation_points:
            for sample_set in batch_set:
                identifier = self.create_settings_file(self.path, sample_set)
                identifier_pool.append(identifier)

        self.dispatch(identifier_pool)
        # Clean pool for next round
        identifier_pool.clear()
        # Launch jobs in cycles as resources open up and then update the model
        current_cycle = 0
        print("Starting Cycles")
        while current_cycle < cycles:
            # Check for complete jobs
            self.completion_status()
            # Get parameters and observation and append to data
            self.update_data()

            if len(self.current_simulations) < (batches * (samples - 1)):
                print("Starting Cycle {}".format(current_cycle))
                model = self.optimizer.train(self.X, self.Y)
                observation_points, _ = self.optimizer.choose(self.bounds, samples=samples, batches=batches, T=T,
                                                              hierarchy=hierarchical)
                chosen_batch = np.random.randint(low=0, high=batches)
                observation_points = observation_points[chosen_batch].reshape(1, samples, -1)
                for batch_set in observation_points:
                    for sample_set in batch_set:
                        identifier = self.create_settings_file(self.path, sample_set)
                        identifier_pool.append(identifier)
                self.dispatch(identifier_pool)
                identifier_pool.clear()
                current_cycle += 1

            time.sleep(10)
        print("Cycles Finished. Waiting on all remaining observation evaluations.")

        # Perform cleanup to wait for everything to complete after cycles are done
        while len(self.current_simulations) > 0:
            # Check for complete jobs
            self.completion_status()
            # Get parameters and observation and append to data
            self.update_data()
            time.sleep(10)

        return self.optimizer.train(self.X, self.Y)

    @staticmethod
    def get_efficiency(path):
        """
        Path to .h5 file with efficiency data. Will check for sufficiently completed simulation and return the efficiency or a preset penalty value of -100
        if the simulation did not complete enough to generate a useful efficiency calculation.

        Return efficiency
        """
        datafile = h5.File(path, 'r')
        if datafile.attrs['complete'] == 0 or datafile.attrs['complete'] == 3:
            # Simulation returned a meaninful result
            efficiency = datafile['efficiency'].attrs['eta']
        else:
            # Simulation did not go far enough to calculate a meaningful efficiency
            # Supply large penalty
            efficiency = -8.

        return efficiency

    @staticmethod
    def get_optimization_parameters(path):
        parameters = yaml.safe_load(open(path, 'r'))
        V = float(parameters['tec']['V_grid'])
        h = float(parameters['tec']['grid_height'])
        r = float(parameters['tec']['strut_height'] / parameters['tec']['strut_width'])

        return V, h, r

    def update_data(self):
        """
        Check `current_simulations` if a value is not None then get the TEC parameters and efficiency and add them to X and Y respectively.
        Then clear the key and value for all completed.
        """
        keys_to_remove = []
        for uuid, eff in self.current_simulations.items():
            if eff:
                V, h, r = self.get_optimization_parameters(os.path.join(self.path, uuid + '.yaml'))
                self.X = np.row_stack([self.X, [V, h, r]])
                self.Y = np.row_stack([self.Y, [eff]])

                keys_to_remove.append(uuid)
        # Cleanup
        for key in keys_to_remove:
            self.current_simulations.pop(key)

    def completion_status(self):
        """
        Look through all `current_simulation` keys and check if there is a directory and .h5 file
        that matches. If found then extract the efficiency and set as value for that key.
        """
        dir_prefix = 'diags_id_'
        file_prefix = 'efficiency_id_'
        for uuid, complete in self.current_simulations.items():
            if not complete:
                directory = dir_prefix + uuid
                filename = file_prefix + uuid + '.h5'
                if os.path.exists(os.path.join(self.path, directory, filename)):
                    self.current_simulations[uuid] = self.get_efficiency(os.path.join(self.path, directory, filename))

    @staticmethod
    def create_settings_file(path, new_settings):
        """
        Make a new TEC description file based on new settings and the 'tec_start.yaml' base file.
        Return the UUID generated, unique identifier for the file.
        """
        base_file = yaml.safe_load(open(os.path.join(path, 'tec_start.yaml'), 'r'))

        V, h, r = new_settings

        base_file['tec']['V_grid'] = float(V)
        base_file['tec']['grid_height'] = float(h)
        base_file['tec']['strut_width'] = float(base_file['tec']['strut_width'] * r)

        identifier = str(uuid.uuid4())
        directory = os.getcwd()
        new_file = identifier + '.yaml'
        yaml.dump(base_file, open(os.path.join(path, new_file), 'w'))

        return identifier
