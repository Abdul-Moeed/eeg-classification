from .dataset import Dataset, DatasetError
import os
import numpy as np
import scipy.io


# TODO: BROKEN!
class Khushaba3(Dataset):
    """A Khushaba2 dataset.

    An Khushaba2 dataset usually consists of three files that are within a specific
    subdirectory. The implementation follows this structuring, i.e. the user
    needs to pass a base-directory as well as the identifier upon instantiation.

    """

    def __init__(self, base_dir, identifier, class_labels=[], **kwargs):
        """Initialize a Khushaba3 dataset without loading it.

        Args:
            base_dir (str): The path to the base directory in which the Khushaba3 dataset resides.
            identifier (str): String identifier for the dataset, e.g. `S1`
            class_labels (list): A list of class labels
            **kwargs: Additional keyword arguments (unused)

        """

        super(Khushaba3, self).__init__(**kwargs)

        self.base_dir = base_dir
        self.data_id = identifier
        self.data_dir = os.path.join(self.base_dir, self.data_id)
        self.data_type = 'EMG'
        self.data_name = 'Khushaba3'

        self._class_labels = ['HandOpen', 'HandRest', 'WristPron', 'WristSupi']

        #self._class2_labels = ['HandOpen', 'HandRest', 'WristPron', 'WristSupi']

        # number of classes in the dataset
        if not isinstance(class_labels, list):
            raise ValueError('Required list of class labels (`class_labels`)')

        self.class_labels = class_labels

        # all Khushaba3 datasets have the same configuration and parameters

        # length of a trial after trial_sample (in seconds)
        self.trial_len = None
        # idle period prior to trial start (in seconds)
        self.trial_offset = None

        # additional variables to store data as expected by the ABC
        self.raw_data_ = None
        self.trials = None
        self.labels = None
        self.sampling_freq = 1000

        self.raw_data = None
        self.duration = 5

        self.trials_  = None
    def load(self, **kwargs):
        """Loads a Khushaba3 dataset.

        For more information about the returned values, see
        :meth:`gumpy.data.Dataset.load`
        """

        self.trials = ()
        self.labels_ = ()
        self.labels = []

        for class_name in self.class_labels:
            classTrials, label_list = self.getClassTrials(class_name)
            self.trials = self.trials + (classTrials,)
            self.labels_ = self.labels_ + (label_list,)

        for i in range(0,4):
            for j in range(0,15):
                self.labels.append(self.labels_[i][j])
        self.labels = np.asarray(self.labels)

        for trial in self.trials:
            if self.raw_data_ is None:
                self.raw_data_ = trial
            else:
                self.raw_data_ = np.concatenate((self.raw_data_, trial))

        self.raw_data =np.array(np.concatenate([np.array(xi) for xi in self.raw_data_]))


        numbers = []
        for i in range(0,4):
            for j in range(0,15):
                numbers.append(len(self.trials[i][j]))

        self.trials_ = [0] * (self.labels.shape[0])
        self.trials_[0]=numbers[0]
        for x in range(1,self.labels.shape[0]):
            self.trials_[x] = sum(numbers[:x+1])

        self.trials = None
        self.trials = []
        self.trials = self.trials_
        self.trials = np.asarray(self.trials)
        
    def getClassTrials(self, class_name):
        """Return all class trials and labels.

        Args:
            class_name (str): The class name for which the trials should be returned

        Returns:
            A 2-tuple containing

            - **trials**: A list of all trials of `class_name`
            - **labels**: A list of corresponding labels for the trials

        """
        Results = []
        label_list = []

        for x in range(1,6):
            for y in range(1,4):
                file = self.base_dir+'{}'.format(class_name)+'/'+'{}'.format(self.data_id)+'/'+'Pos{}_{}_-{}.mat'.format(x, class_name, y)

                trial = scipy.io.loadmat(file)['data'][:,:7]
                trial = scipy.signal.decimate(trial,4, n=None, ftype='iir', axis=0, zero_phase=True)

                Results.append(trial)
                label_list.append(self._class_labels.index(class_name))

        Results=np.asarray(Results)
        label_list=np.asarray(label_list)
        return Results, label_list
