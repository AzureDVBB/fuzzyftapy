"""
A python scrypt to do Fuzzy Fault Tree Analysis.
"""

### Licence notice (MIT)
__licence__ = """
Copyright (c) 2021 Benjamin Bence Végh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# module import statements ##########################################

import math                   # built-in mathematical functions
import os                     # tools for interacting with OS, namely file paths
import json                   # used to load and save tree to disk
import argparse               # used to implement the Command-Line Interface
import logging                # warning and error logs
import itertools              # utility functions for use with iterables
import time                   # timing execution speed mostly

# module level variables ############################################
__PRECISION__ = 12              # number of digits PRECISION to avoid floating point errors
__version__ = "0.0.1"           # program VERSION
__MINIMUM_VERSION__ = "0.0.1"   # minimal supported program VERSION

# exception definitions #############################################
class FaultTreeLoadError(Exception):
    """An issue has occurred during file loading."""

class FaultTreeError(Exception):
    """An issue has occoured during some operation of the Fault Tree."""

class ComputationOrderingError(Exception):
    """An issue has occurred during the calculation of computation order."""

class VersionError(Exception):
    """An issue has occurred during version checking."""

class FuzzyNumberError(Exception):
    """An issue has occoured during the running of fuzzy number class operations."""

# probability logic operations ######################################
class ProbabilityTools:
    """A collection of functions to work with probabilities."""
    
    @staticmethod
    def logical_and(probabilities: list[float]):
        """
        Performs the AND logic operation on a list of probability numbers.
        """

        if not all([isinstance(i, float) for i in probabilities]):
            raise TypeError(f"Not all of the supplied probabilities were floating point numbers. '{probabilities}'")

        return round(math.prod(probabilities), __PRECISION__)

    @staticmethod
    def logical_or(probabilities: list[float]):
        """
        Performs the OR logic operation on a list of probability numbers.
        """

        if not all([isinstance(i, float) for i in probabilities]):
            raise TypeError(f"Not all of the supplied probabilities were floating point numbers. '{probabilities}'")

        return round(1 - math.prod([1-p for p in probabilities]), __PRECISION__)

    @staticmethod
    def logical_not(probability):
        """
        Performs the NOT logic iperation on a single probability number
        """

        if not all([isinstance(i, float) for i in probabilities]):
            raise TypeError(f"Not all of the supplied probabilities were floating point numbers. '{probabilities}'")

        return round(1 - probability, __PRECISION__)


# data holding object classes #######################################
class AlphaLevelInterval:
    """
    Object to hold the result of an alpha-cut to a trapezoid or triangular fuzzy number.
    """

    # typehints for class attributes
    lower : float
    upper : float
    alphaLevel : float

    def __init__(self, lower: float, upper: float, alphaLevel: float):

        # check and force inputs to be floating point
        try:
            lower = float(lower)
            upper = float(upper)
            alphaLevel = float(alphaLevel)
        
        except ValueError:
            raise ValueError(f"Supplied inputs to the AlphaLevelInterval were not floating point. {lower, upper, alphaLevel}")

        # check if supplied inputs were in the correct range
        if not 0 <= lower <= upper <= 1:
            raise ValueError(f"Lower and Upper bounds must conform to '0 <= lower <= upper <= 1' rule. ({lower}, {upper})")

        if not 0 <= alphaLevel <= 1:
            raise ValueError(f"Alpha level must conform to '0 <= alphaLevel <= 1' rule but '{alphaLevel}' was supplied.")

        self.lower = lower
        self.upper = upper
        self.alphaLevel = alphaLevel
    
    def __repr__(self):
        return f"AlphaLevelInterval(lower={self.lower}, upper={self.upper}, alphaLevel={self.alphaLevel})"
    
    def to_list(self):
        """Returns the list representation of the interval."""
        return [self.lower, self.upper, self.alphaLevel]

    @staticmethod
    def logical_and(terms: list):
        """
        Performs the logical AND operation on a list of 'AlphaLevelInterval'
        """

        # check if supplied terms were all of compatible type
        if not all([isinstance(t, AlphaLevelInterval) for t in terms]):
            raise TypeError("Only 'AlphaLevelInterval' objects are supported, while other types were supplied in the list.")

        lower = ProbabilityTools.logical_and([t.lower for t in terms])
        upper = ProbabilityTools.logical_and([t.upper for t in terms])

        return AlphaLevelInterval(lower, upper, terms[0].alphaLevel)

    @staticmethod
    def logical_or(terms: list):
        """
        Performs the logical OR operation on a list of 'AlphaLevelInterval'
        """

        # check if supplied terms were all of compatible type
        if not all([isinstance(t, AlphaLevelInterval) for t in terms]):
            raise TypeError("Only 'AlphaLevelInterval' objects are supported, while other types were supplied in the list.")

        lower = ProbabilityTools.logical_or([t.lower for t in terms])
        upper = ProbabilityTools.logical_or([t.upper for t in terms])

        return AlphaLevelInterval(lower, upper, terms[0].alphaLevel)


class TrapezoidalFuzzyNumber:
    """
    Fuzzy number object that has a trapezoidal shape.
    """

    # typehints ######
    x1 : float
    x2 : float
    x3 : float
    x4 : float
    
    # internal class variables ######
    _CRISP_STRATEGY: str = None # strategy to convert the fuzzy number to a crisp number, None means it's not crisp
    _IMPLEMENTED_CRISP_STRATEGIES = ['interval-x2x3', 'fully-available', 'fully-unavailable']

    # object constructors #############
    def __init__(self, x1: float, x2: float, x3: float, x4: float):
        
        # check and force inputs to be floating point
        try:
            x1 = float(x1)
            x2 = float(x2)
            x3 = float(x3)
            x4 = float(x4)
        
        except ValueError:
            raise FuzzyNumberError(f"Supplied inputs to the TrapezoidalFuzzyNumber were not floating point. {x1, x2, x3, x4}")

        # verify that inputs conform to the rule
        if not 0 <= x1 <= x2 <= x3 <= x4 <= 1:
            raise FuzzyNumberError(f"The trapezoidal fuzzy number must conform to '0 <= x1 <= x2 <= x3 <= x4 <= 1' rule. But '{[x1, x2, x3, x4]}' was supplied.")

        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
    
    @staticmethod
    def from_list(terms: list):
        """Constructs the Trapezoid Fuzzy Number object from a list of 4 key points."""

        if len(terms) != 4:
            raise FuzzyNumberError(f"Keypoint count is not 4 while loading from list (trapezoidal). '{terms}'")

        return TrapezoidalFuzzyNumber(terms[0], terms[1], terms[2], terms[3])

    @staticmethod
    def from_list_triangular_with_errorfactor(terms: list):
        """Constructs the Trapezoid Fuzzy Number object from a list of 2 elements, the median value and error factor."""

        if len(terms) != 2:
            raise FuzzyNumberError(f"Keypoint count is not 2 while loading from list (triangular with error factor). '{terms}'")

        x1 = round(terms[0] / terms[1], __PRECISION__)
        x2 = x3 = terms[0]
        x4 = round(terms[0] * terms[1], __PRECISION__)

        return TrapezoidalFuzzyNumber(x1, x2, x3, x4)
    
    @staticmethod
    def from_list_triangular(terms: list):
        """Constructs the Trapezoid Fuzzy Number object from a list of 3 keypoints."""

        if len(terms) != 3:
            raise FuzzyNumberError(f"Keypoint count is not 3 while loading from list (triangular). '{terms}'")
        
        return TrapezoidalFuzzyNumber(terms[0], terms[1], terms[1], terms[2])
    
    # text representation of the class ####
    def __repr__(self):
        return f"TrapezoidalFuzzyNumber(x1 = {self.x1}, x2 = {self.x2}, x3 = {self.x3}, x4 = {self.x4})"

    # string representation of the class, helps with saving ####
    def __str__(self):
        return str([self.x1, self.x2, self.x3, self.x4])
    
    # export to list form ####
    def to_list(self):
        return [self.x1, self.x2, self.x3, self.x4]

    # alphacutting #####
    def alphacut(self, alphaLevel: float):
        """
        Performs an alpha-cut on the Fuzzy Number at the specified alpha-levl, returning an interval.
        """

        # typechecking to the alphaLevel
        try:
            alphaLevel = float(alphaLevel)
        except ValueError:
            raise FuzzyNumberError(f"Supplied alpha level was not a number. '{alphaLevel}'")

        # check if alpha level is within bounds
        if not 0 <= alphaLevel <= 1:
            raise FuzzyNumberError("Alpha level must be between 1 and 0 inclusive.")
        
        # do the alpha cut with the designated strategy
        if not self._CRISP_STRATEGY is None: # replace self with a TrapezoidFuzzyNumber with correct strategy if there is one used
            self = self.apply_crisp_strategy()

        # perform alphacut
        lower = round(self.x1 + alphaLevel * (self.x2 - self.x1), __PRECISION__)
        upper = round(self.x4 - alphaLevel * (self.x4 - self.x3), __PRECISION__)

        return AlphaLevelInterval(lower, upper, alphaLevel)
    
    # logical operations ##########
    @staticmethod
    def logical_and(terms: list):
        """Performs the logical AND operation on a list of 'AlphaLevelInterval'."""

        if not all([isinstance(t, TrapezoidalFuzzyNumber) for t in terms]):
            raise FuzzyNumberError("Only 'TrapezoidalFuzzyNumber' objects are supported, while other types were supplied in the list.")

        # apply crisp strategy to all supplied trapezoidal fuzzy numbers
        terms = [t.apply_crisp_strategy() for t in terms]

        # do the pointwise logical operation
        x1 = ProbabilityTools.logical_and([t.x1 for t in terms])
        x2 = ProbabilityTools.logical_and([t.x2 for t in terms])
        x3 = ProbabilityTools.logical_and([t.x3 for t in terms])
        x4 = ProbabilityTools.logical_and([t.x4 for t in terms])

        return TrapezoidalFuzzyNumber(x1, x2, x3, x4)

    @staticmethod
    def logical_or(terms: list):
        """Performs the logical OR operation on a list of 'AlphaLevelInterval'."""

        if not all([isinstance(t, TrapezoidalFuzzyNumber) for t in terms]):
            raise FuzzyNumberError("Only 'TrapezoidalFuzzyNumber' objects are supported, while other types were supplied in the list.")

        # apply crisp strategy to all supplied trapezoidal fuzzy numbers
        terms = [t.apply_crisp_strategy() for t in terms]

        # do the pointwise logical operation
        x1 = ProbabilityTools.logical_or([t.x1 for t in terms])
        x2 = ProbabilityTools.logical_or([t.x2 for t in terms])
        x3 = ProbabilityTools.logical_or([t.x3 for t in terms])
        x4 = ProbabilityTools.logical_or([t.x4 for t in terms])

        return TrapezoidalFuzzyNumber(x1, x2, x3, x4)

    # defuzzification
    def apply_crisp_strategy(self):
        """Returns a new TrapezoidalFuzzyNumber with the defuzzification strategy applied to it."""

        if self._CRISP_STRATEGY is None:
            return self

        elif self._CRISP_STRATEGY == "fully-available":
                return TrapezoidalFuzzyNumber(0, 0, 0, 0)

        elif self._CRISP_STRATEGY == "fully-unavailable":
            return TrapezoidalFuzzyNumber(1, 1, 1, 1)

        elif self._CRISP_STRATEGY == "interval-x2x3":
            return TrapezoidalFuzzyNumber(self.x2, self.x2, self.x3, self.x3)
        
        else: # supplied strategy was for some reason not implemented
            raise FuzzyNumberError(f"Invalid strategy for defuzzification on the TrapezoidalFuzzyNumber during apply."
                                   f" '{self._CRISP_STRATEGY}' was supplied while only the following are"
                                   f" implemented for this class: '{self._IMPLEMENTED_CRISP_STRATEGIES}'"
                                   )

    # convenience methods to set defuzzification method
    def set_crispness_to_fully_avalable(self):
        """Set Fuzzy Number to behave as a crisp value that is fully available (all points set to 0)."""
        self._CRISP_STRATEGY = "fully-available"

    def set_crispness_to_fully_unavalable(self):
        """Set Fuzzy Number to behave as a crisp value that is fully unavailable (all points set to 1)."""
        self._CRISP_STRATEGY = "fully-unavailable"
    
    def set_crispness_to_interval_x2x3(self):
        """Set Fuzzy Number to behave as a crisp value that is an interval (square shape number) between points x2, x3."""
        self._CRISP_STRATEGY = "interval-x2x3"
    
    def reset_crispness(self):
        """Reset Fuzzy Number behavior to NOT be crisp anymore."""
        self._CRISP_STRATEGY = None

# main Fault tree class definition ##################################
class FuzzyFaultTree:

    # typehints
    metadata : dict
    baseEvents : dict
    logicGates : dict

    # class variables for calculations ##############################
    _MAX_SEARCH_DEPTH = 1000

    # definitions for file integrity checks #########################
    _METADATA_STRUCTURE = {"version": 'str', # typecheck
                           "base-event-shape": ["trapezoidal", "triangular", "triangular-errorfactor"] # exact values it can take
                           }

    _LOGIC_GATE_STRUCTURE = {"type": ["and", "or"],
                             "inputs": 'list'
                             }

    # initialization ################################################
    def __init__(self, baseEventShape: str = 'trapezoidal', topEventGateType: str = 'and'):
        """Instance an empty fuzzy fault tree"""
        self.metadata = {'version': __version__, 'base-event-shape': baseEventShape}
        self.baseEvents = {}
        self.logicGates = {'top-event': {'type': topEventGateType, 'inputs': []}}
    
    # checking and verification methods #############################
    @staticmethod
    def version_check(version: str):
        """Verifies that the version of the fuzzy fault tree is acceptable, returning True if it is."""

        if not isinstance(version, str):
            raise VersionError("Supplied version to check is not of type string.")

        # split the version numbers seperated by dots into a list of individual versions
        minVersionParts = __MINIMUM_VERSION__.split('.')
        thisVersionParts = version.split('.')
        maxVersionParts = __version__.split('.')
    
        if not len(thisVersionParts) == len(maxVersionParts) == len(minVersionParts):
            raise VersionError(f"Supplied version, max version or minimal version contains different number of parts. (dot separated)"
                               f" MinVersion: '{__MINIMUM_VERSION__}' , MaxVersion: '{__version__}' , suppliedVersion: '{version}'")

        # get the longest string length of all list elements
        versionNumberLength = max([len(vnum) for vnum in minVersionParts + maxVersionParts + thisVersionParts])
        
        # extend and join each part to form a single version number
        minVersion = ''.join([part.ljust(versionNumberLength, '0') for part in minVersionParts])
        thisVersion = ''.join([part.ljust(versionNumberLength, '0') for part in thisVersionParts])
        maxVersion = ''.join([part.ljust(versionNumberLength, '0') for part in maxVersionParts])

        # compare version numbers and return the answer if its okay or not
        return minVersion <= thisVersion <= maxVersion
        
    # fault tree saving and loading to disk #########################
    @staticmethod
    def load_from_file(loadPath : str):

        if not os.path.isfile(loadPath):
            raise FaultTreeLoadError(f"No file found at path: '{loadPath}'")

        treeDict = None

        with open(loadPath) as f:
            treeDict = json.load(f)

        # verify metadata is correct and version is supported
        metadata = treeDict['metadata']
        if not all([k in FuzzyFaultTree._METADATA_STRUCTURE.keys() for k in metadata.keys()]):
            raise FaultTreeLoadError(f"During file loading the METADATA section did "
                                     f"not conform to the following structure: {FuzzyFaultTree._METADATA_STRUCTURE}")

        if not FuzzyFaultTree.version_check(metadata['version']):
            raise FaultTreeLoadError(f"Unsupported version: '{metadata['version']}' ."
                                     f"version support is from '{__MINIMUM_VERSION__}' to '{__version__}' .")

        if not metadata['base-event-shape'] in FuzzyFaultTree._METADATA_STRUCTURE['base-event-shape']:
            raise FaultTreeLoadError(f"Unsupported base event shape."
                                     f" Shape must be any one of the following strings (all lower case): "
                                     f"{FuzzyFaultTree._METADATA_STRUCTURE['base-event-shape']}")

        # decode base events into correct objects
        baseEvents = treeDict['base-events']

        if metadata['base-event-shape'] == "trapezoidal":
            # replace all values in the dict with actual objects
            for name, event in baseEvents.items():
                baseEvents[name] = TrapezoidalFuzzyNumber.from_list(event)
        elif metadata['base-event-shape'] == "triangular-errorfactor":
            # replace all values in the dict with actual objects
            for name, event in baseEvents.items():
                baseEvents[name] = TrapezoidalFuzzyNumber.from_list_triangular_with_errorfactor(event)
        elif metadata['base-event-shape'] == "triangular":
            # replace all values in the dict with actual objects
            for name, event in baseEvents.items():
                baseEvents[name] = TrapezoidalFuzzyNumber.from_list_triangular(event)
        else:
            raise FaultTreeLoadError(f"Base event shape '{metadata['base-event-shape']}' not supported.")

        # load logic gate connections and check correctness
        logicGates = treeDict['logic-gates']
        for name, gateData in logicGates.items():
            if not all([k in FuzzyFaultTree._LOGIC_GATE_STRUCTURE.keys() for k in gateData.keys()]):
                raise FaultTreeLoadError(f"Logic gate '{name}' does not conform to the structure: "
                                         f"{FuzzyFaultTree._LOGIC_GATE_STRUCTURE}")

            for event in gateData['inputs']:
                if not (event in baseEvents or event in logicGates):
                    raise FaultTreeLoadError(f"Logic gate '{name}' has input event '{event}' "
                                             f"which is neither a Base Event or an Intermediate event."
                                             f"(doesn't appear in either 'logic-gates' or 'base-events' as a key)")

            if gateData['type'] in FuzzyFaultTree._LOGIC_GATE_STRUCTURE['type']:
                FaultTreeLoadError(f"Logic gate '{name}' has unsupported type."
                                   f" Logic gate types must be one of: "
                                   f"'{FuzzyFaultTree._LOGIC_GATE_STRUCTURE['type']}'")

            if name in baseEvents:
                raise FaultTreeLoadError(f"Logic gate name '{name}' already used for a base event. "
                                         f"Logic gate names must be unique and cannot be a base even name.")

        # construct tree object and fill it with needed data

        fft = FuzzyFaultTree()
        fft.metadata = metadata
        fft.baseEvents = baseEvents
        fft.logicGates = logicGates

        return fft

    def save_to_file(self, savePath: str):
        """Saves the fault tree to disk in a JSON file."""

        if os.path.exists(savePath):
            raise FaultTreeError(f"Could not save fault tree, destination is a file or folder / '{savePath}'")
        
        # serialize base events
        md = self.metadata.copy()
        be = {n: str(self.baseEvents[n]) for n in self.baseEvents.keys()}
        lg = self.logicGates.copy()

        with open(savePath, 'w') as f:
            json.dump({'metadata': md, 'base-events': be, 'logic-gates': lg}, f, indent=4)

    # methods to manipulate the fuzzy fault tree structure ##########
    def api_add_logic_gate(self, parentGate: str, name: str, gateType: str, inputs: list = []):
        """
        Adds a single logic gate to the given parent (an existing logic gate where it will be added as an input to it).
        """

        if not parentGate in self.logicGates:
            raise FaultTreeError(f"Cannot create new logic gate '{name}': could not find existing gate '{parentGate}' to set as parent.")
        
        elif name in self.logicGates or name in self.baseEvents:
            raise FaultTreeError(f"Cannot create new logic gate '{name}': name already exists as a base event or logic gate.")
        
        elif not gateType in self._LOGIC_GATE_STRUCTURE['type']:
            raise FaultTreeError(f"Cannot create new logic gate '{name}': supplied type '{gateType}' is not implemented.")
        
        # ensure it doesn't already exist as an input for some reason
        if not name in self.logicGates[parentGate]["inputs"]:
            self.logicGates[parentGate]["inputs"].append(name) # update parent gate with it being an input
        
        self.logicGates[name] = {"type": gateType, "inputs": inputs} # register new gate
    
    def api_remove_logic_gate(self, gateName: str):
        """Removes an existing logic gate from the tree. WARNING! Removes all logic gates bellow it."""

        if not gateName in self.logicGates:
            raise FaultTreeError(f"There is no logic gates in the tree with name '{gateName}'.")

        
        # search which gate is it's parent and remove from inputs
        for gate in self.logicGates:
            if gateName in self.logicGates[gate]["inputs"]:
                self.logicGates[gate]["inputs"].remove(gateName)
        
        # delete gate and dependancies from tree
        dependancies = [gateName]
        dependancies.extend(self._calculate_dependent_logic_gates(gateName))
        for d in dependancies:
            del self.logicGates[d]

    def api_add_base_event(self, name: str, fuzzyNumber):
        """Adds a fuzzy number to the tree as a new base event."""

        if name in self.logicGates or name in self.baseEvents:
            raise FaultTreeError(f"Could not add Base event '{name}' : it already exists as a base event or logic gate.")
        
        # hardcoded check for trapezoidal fuzzy number
        if self.metadata["base-event-shape"] in ["trapezoidal", "triangular", "triangular-errorfactor"] and not isinstance(fuzzyNumber, TrapezoidalFuzzyNumber):
            raise FaultTreeError(f"Could not add base event '{name}' : the fault tree's base event shape is trapezoidal or triangular"
                                 f" but the supplied fuzzy number was not of type TrapezoidalFuzzyNumber.")
        
        self.baseEvents[name] = fuzzyNumber

    def api_remove_base_event(self, name: str):
        """Remove specified base event and all occourences in logic gate inputs."""

        if not name in self.baseEvents:
            raise FaultTreeError(f"Could not find base event '{name}' to attempt to remove.")
        
        # search and remove from all logic gate inputs
        for gate in self.logicGates:
            try:
                self.logicGates[gate]["inputs"].remove(name)
            except ValueError: # ignore value errors that arise from input list not containing it
                pass
        
        del self.baseEvents[name]

    def api_assign_base_event_to_gate_input(self, eventName: str, gateName: str):
        """Adds an existing base event to a specified logic gate's inputs."""

        if not eventName in self.baseEvents:
            raise FaultTreeError(f"Could not assing base event '{eventName}' :  no base event with that name exists.")
        
        elif not gateName in self.logicGates:
            raise FaultTreeError(f"Could not assing base event '{eventName}' :  no logic gate with name '{gateName}' exists.")
        
        if not eventName in self.logicGates[gateName]["inputs"]:
            self.logicGates[gateName]["inputs"].append(eventName)

    def api_remove_input_from_gate(self, inputName: str, gateName: str):
        """
        Removes an existing input from a logic gate's inputs. WARNING! If the input is a logic gate, 
        all logic gates bellow will also be removed.
        """
        
        if not (inputName in self.logicGates or inputName in self.baseEvents):
            raise FaultTreeError(f"Could not unassing input event '{inputName}' :  no logic gate with name '{gateName}' exists.")
        
        if inputName in self.baseEvents: # the input is a base event, merely remove from list
            try:
                self.logicGates[gateName]["inputs"].remove(inputName)
            except ValueError: # ignore error from the inputs not containing the entry
                pass
        
        elif inputName in self.logicGates: # input is a logic gate
            self.api_remove_logic_gate(inputName)

        else:
            raise FaultTreeError(f"Could not unassign input event '{eventName}'")

    # calculations and sub-calculations #############################
    def _calculate_computation_order(self):
        """
        Calculates the order in which computation has to take place, starting from base events to top event.
        This function can also be used to determine which calculations can be parallelized.
        Returns a list of keys from both base events and logic gates.
        """

        # first order of operations is base event computation to simplify things
        processed = list(self.baseEvents.keys())
        computeOrder = [list(self.baseEvents.keys())]

        allEventKeys = []
        allEventKeys.extend(self.baseEvents.keys())
        allEventKeys.extend(self.logicGates.keys())

        # outer loop for how many times to iterate through the tree to find the correct compute order
        for _ in range(self._MAX_SEARCH_DEPTH):
            # finished successfully
            if len(processed) == len(allEventKeys):
                break

            # main compute order deciding loop
            currentComputeOrder = []
            currentProcessed = []

            for name, gate in self.logicGates.items():
                # event key already processed, skipping
                if name in processed:
                    continue

                # an input event is neither base nor intermediate event
                elif not all([i in allEventKeys for i in gate['inputs']]):
                    raise ComputationOrderingError(f"Gate '{name}' has inputs that are neither another gate or a base event")

                # all input events have been computed already
                elif all([i in processed for i in gate['inputs']]):
                    currentProcessed.append(name)
                    currentComputeOrder.append(name)
                    continue

                # one or more input events have not been computed at this iteration
                # this line is not needed, however was included to make it more understandable
                else:
                    continue

            # add this iteration's compute order
            computeOrder.append(currentComputeOrder)
            processed.extend(currentProcessed)

        else: # did not successfully finish setting the compute order in given loop count
            raise ComputationOrderError("Maximum iteration count reached.")

        return computeOrder
    
    def _calculate_dependent_logic_gates(self, gateName: str):
        """Calculates the logic gates bellow the given logicGate that it depends on directly or indirectly."""

        if not gateName in self.logicGates:
            raise FaultTreeError(f"Could not calculate dependencies of '{logicGate}' :  no logic gate with name exists.")

        # get the direct dependancies and filter the logic gates
        search = [i for i in self.logicGates[gateName]["inputs"] if i in self.logicGates]
        dependancies = search.copy() # working copy of the dependancies, the rest will be added here too

        for _ in range(self._MAX_SEARCH_DEPTH):
            # search finished successfully
            if search == []:
                return dependancies
            
            # search and filter for logic gate dependancies in the next logic gate in the search list
            found = [i for i in self.logicGates[search[0]]["inputs"] if i in self.logicGates]

            # add the found dependancies to both lists
            dependancies.extend(found)
            search.extend(found)

            # pop the processed element off the list of logic gates to search
            search.pop(0)

        else:
            raise FaultTreeError(f"Could not calculate dependencies of '{logicGate}' : MAX_SEARCH_DEPTH reached without success.")

    def calculate_minimum_cut_sets(self, newTree: bool = False, analyzedGate='top-event', _topCall = True):
        """
        Calculate the minimum cut sets of the fuzzy fault tree, returning a list of lists which is 
        the Sum of Products (SOP) representation of the fault tree.
        
        If the newTree is set 'True' it will instead return a new fuzzy fault tree, where
        the logic gates are replaced to be the Sum of Products representation.

        The analyzedGate input is used to determine which logic gate to determine the minimum cut
        cut sets for with the gates and events bellow it.

        WARNING!!! This is a specialized function, not a generic one.
        It may need to be replaced later on when new gate types are added.

        MEGAWARNING!!! This is a recursive function, the _topCall variable should not be used ever
        as it is used in the recursion. Also maximum recursion depth can be reached in extreme cases.
        """

        gateType = self.logicGates[analyzedGate]["type"]
        inputs = self.logicGates[analyzedGate]["inputs"]

        visited = []

        # I barely know how this works, but esentially recursive node visits
        for i in inputs:
            # node is base event so we just add it to the visited pile
            if i in self.baseEvents:
                visited.append([i])

            # node is logic gate so we call ourselves to get all base events, recursively
            else:
                visited.append(self.calculate_minimum_cut_sets(analyzedGate=i, _topCall = False))
        
        # the node analyzed is type AND, we merge the visited into a single list and return it
        if gateType == "and":
            ret = list(itertools.product(*visited))

        # the node analyzed is type OR we just string together the list and return it
        elif gateType == "or":
            ret = list(itertools.chain(*visited))
        
        if not _topCall: # ensure recursiveness finishes
            answer = ret

        # top call reached, finish up and return answer
        else: 

            # very naughty recursive way to flatten deeply nested lists
            def flatten(L, outerLayer = False):
                if not (isinstance(L, list) or isinstance(L, tuple)):
                    return [L]
                else: 
                    if outerLayer:
                        return list(itertools.chain([flatten(x) for x in L]))
                    else:
                        return list(itertools.chain(*[flatten(x) for x in L]))
            
            flt = flatten(ret, outerLayer=True)

            # we want the result to just be a list of minimum cut sets
            if not newTree:
                answer = flt
            
            # we want a new fault tree that has the logic gates replaced by minimum cut sets as a result
            else:
                # create a copy of the fault tree
                ft = FuzzyFaultTree(self.metadata["base-event-shape"], "or")
                ft.baseEvents = self.baseEvents.copy()
                ft.metadata = self.metadata.copy()

                # add all the minimum cut sets in SOP representation
                for i, inputs in enumerate(flt):
                    ft.api_add_logic_gate('top-event', f'MCS-{i+1}', 'and', inputs)
                
                answer = ft
        
        return answer

    def calculate_top_event(self, useMinCutSets: bool = False):
        """Calculates the top-event fuzzy number."""

        if useMinCutSets:
            self = self.calculate_minimum_cut_sets(newTree=True)
        
        computeOrder = self._calculate_computation_order()
        computed = {}

        for depth, layerEvents in enumerate(computeOrder):
            for event in layerEvents:
                if depth == 0:
                    computed[event] = self.baseEvents[event]
            
                elif self.logicGates[event]['type'] == "and":
                    computed[event] = TrapezoidalFuzzyNumber.logical_and([computed[e] for e in self.logicGates[event]['inputs']])

                elif self.logicGates[event]['type'] == "or":
                    computed[event] = TrapezoidalFuzzyNumber.logical_or([computed[e] for e in self.logicGates[event]['inputs']])
                
                else:
                    raise Exception("logic gate neither 'and' nor 'or' ")
        
        return computed['top-event']

    def calculate_top_event_alphacut(self, alphaLevel: float, useMinCutSets: bool = False):
        """Calculates the Alpha-Cut of the top event at the given alpha-level"""

        if useMinCutSets:
            self = self.calculate_minimum_cut_sets(newTree=True)
        
        computeOrder = self._calculate_computation_order()
        computedCuts = {}
        
        for layer in computeOrder:
            for event in layer:
                if event in self.baseEvents:
                    computedCuts[event] = self.baseEvents[event].alphacut(alphaLevel)
                else:
                    gate = self.logicGates[event]
                    if gate['type'] == "and":
                        computedCuts[event] = AlphaLevelInterval.logical_and([computedCuts[k] for k in gate['inputs']])

                    elif gate['type'] == "or":
                        computedCuts[event] = AlphaLevelInterval.logical_or([computedCuts[k] for k in gate['inputs']])
                    else:
                        raise Exception("gate type is neither 'and' , 'or' ")

        return computedCuts['top-event']

    def calculate_fim(self, alphaLevels=[0.0], useMinCutSets: bool = False):
        """
        Calculates the fuzzy importance measure of each base event at a given alpha level.
        Then assigns a ranking to each base event.
        """
        
        if useMinCutSets:
            self = self.calculate_minimum_cut_sets(newTree=True)

        # fully 1 or 0 base event probability -> top event probability
        fim = {}
        
        # calculate for each alpha level and each event, the top event probability when specified base event is always 0 or 1
        for eventName in self.baseEvents:
            currentFim = 0
            for level in alphaLevels:
                qi = []

                # top event with base event fully available
                self.baseEvents[eventName].set_crispness_to_fully_avalable()
                qi.append(self.calculate_top_event_alphacut(level))

                # top event with base event fully unavailable
                self.baseEvents[eventName].set_crispness_to_fully_unavalable()
                qi.append(self.calculate_top_event_alphacut(level))

                # reset base event to normal
                self.baseEvents[eventName].reset_crispness()

                currentFim += round(math.sqrt((qi[0].lower - qi[1].lower)**2 + (qi[0].upper - qi[1].upper)**2), __PRECISION__)
        
            fim[eventName] = round(currentFim, __PRECISION__)

        # rank FIM by sorting the dictionary by keys and enumerating the result, ignoring the values
        fimRanks = {n: {"rank": None, "value": v} for n, v in fim.items()}

        lastNumber = None
        offset = 1
        for rank, name in enumerate([k for k, _ in sorted(fim.items(), key=lambda item: item[1], reverse=True)]):
            if fimRanks[name]["value"] == lastNumber: offset -= 1
            fimRanks[name]["rank"] = rank + offset
            lastNumber = fimRanks[name]["value"]

        return fimRanks
        
    def calculate_fuim(self, alphaLevels=[0.0], defuzzingMethod='interval-x2x3', useMinCutSets: bool = False):
        """
        Calculates the fuzzy uncertainty importance measure of each base event at a given alpha level.
        Then assigns a rank to each base event.
        """

        if useMinCutSets:
            self = self.calculate_minimum_cut_sets(newTree=True)
        
        # fully 1 or 0 base event probability -> top event probability
        fuim = {}
        
        # calculate top event normally for each alpha level
        q = [self.calculate_top_event_alphacut(level) for level in alphaLevels]

        # calculate for each alpha level and each event, the top event probability when specified base event is always 0 or 1
        for eventName in self.baseEvents:
            
            # calculate top event with analyzed event being crisp in value for each alpha level
            self.baseEvents[eventName]._CRISP_STRATEGY = defuzzingMethod
            qi = [self.calculate_top_event_alphacut(level) for level in alphaLevels]
            self.baseEvents[eventName].reset_crispness()

            # calculate euclidean distance of the two top events at each alpha level and then sum the results
            euclideanDistances = [math.sqrt((q[x].lower - qi[x].lower)**2 + (q[x].upper - qi[x].upper)**2) for x in range(len(alphaLevels))]
            fuim[eventName] = round(sum(euclideanDistances), __PRECISION__)

        # rank FIM by sorting the dictionary by keys and enumerating the result, ignoring the values
        fuimRanks = {n: {"rank": None, "value": v} for n, v in fuim.items()}

        lastNumber = None
        offset = 1
        for rank, name in enumerate([k for k, _ in sorted(fuim.items(), key=lambda item: item[1], reverse=True)]):
            if fuimRanks[name]["value"] == lastNumber: offset -= 1
            fuimRanks[name]["rank"] = rank + offset
            lastNumber = fuimRanks[name]["value"]

        return fuimRanks

# graphing ##########################################################

def visualize_fft(fft: FuzzyFaultTree, alphaLevelsTopEvent=[0, 1], alphaLevelsMeasures=[0, 1], topEvent: bool = False, measures: bool = False, useMinCutSets: bool = False):
    from matplotlib import pyplot as plt

    if useMinCutSets:
        fft = fft.calculate_minimum_cut_sets(newTree = True)

    if topEvent:
        fig, ax1 = plt.subplots()

        cuts = {}
        cuts['traditional-estimate'] = [fft.calculate_top_event().alphacut(level) for level in alphaLevelsTopEvent]
        cuts['alphacut'] = [fft.calculate_top_event_alphacut(level) for level in alphaLevelsTopEvent]

        yAxes = alphaLevelsTopEvent.copy()
        yAxes.extend(reversed(alphaLevelsTopEvent))
        xAxes = {}
        for name in cuts.keys():
            xAxes[name] = [(cuts[name][i].lower) for i in range(len(alphaLevelsTopEvent))]
            xAxes[name].extend([(cuts[name][i].upper) for i in reversed(range(len(alphaLevelsTopEvent)))])


        ax1.plot(xAxes['traditional-estimate'], yAxes, label='traditional-estimate', linewidth=0.5, color='red')
        ax1.plot(xAxes['alphacut'], yAxes, label='alphacut', linewidth=0.5, color='black')

        ax1.set_ylabel("Degree of Membership [μ]")
        ax1.set_xlabel("Top Event Probability [x]")
        ax1.yaxis.grid() # set horizontal grid lines
        ax1.set_yticks(alphaLevelsTopEvent) # set y ticks to discreet levels
        ax1.set_ylim(0, 1.05)
        ax1.legend()
    
    if measures:
        fig, ax2 = plt.subplots()

        fim = fft.calculate_fim(alphaLevelsMeasures)
        fuim = fft.calculate_fuim(alphaLevelsMeasures)

        # TODO: make work
        fimRanks = [fim[k]['rank'] for k in fim.keys()]
        fuimRanks = [fuim[k]['rank'] for k in fim.keys()]

        barWidth = 0.3

        ax2.bar([x - barWidth/2 for x in range(len(fim.keys()))], fimRanks, barWidth, label="FIM Ranks", color="blue")
        ax2.bar([x + barWidth/2 for x in range(len(fim.keys()))], fuimRanks, barWidth, label="FUIM Ranks", color="red")

        ax2.yaxis.grid() # set horizontal grid lines
        ax2.set_yticks(range(len(fim.keys())+1))
        ax2.set_ylabel("Importance Measure Rank")
        ax2.set_xticks(range(len(fim.keys())))
        ax2.set_xticklabels(fim.keys(), rotation=90)
        ax2.legend()

    plt.show()


if __name__ == "__main__":
    # this section is used when using this software standalone

    # implement command line interface
    import argparse
    import pprint
    pp = pprint.PrettyPrinter(indent=4)

    # parser to handle argument inputs and help
    parser = argparse.ArgumentParser(description="Fuzzy Fault Tree Analysis software.")

    # argument groups
    frontendGroup = parser.add_mutually_exclusive_group(required=True)
    frontendGroup.add_argument('-gui', action='store_true', help="Launches the GUI frontend.")
    frontendGroup.add_argument('-i', '--input-file', type=str, help="Specifies the input JSON file that has the fault tree.")

    alphaLevelGroup = parser.add_mutually_exclusive_group()
    alphaLevelGroup.add_argument('-l', '--alpha-levels', type=float, nargs="+", default=[0.0, 1.0], 
                                 help="Makes the calculations use the specified alpha levels.")
    alphaLevelGroup.add_argument('-nl', '--number-of-alpha-levels', type=int,
                                 help="""Makes calculations use a list of alpha levels equally divided between 0 and 1 a number of times supplied. (minimum of 2)""")

    # define all arguments here (that do not)
    parser.add_argument('-o', '--output-file', type=str, help="Save results to a new JSON file. (You are responsible for the file extension.)")
    parser.add_argument('-v', '--visualize', action='store_true', 
                        help="Visualize the computation results using 'matplotlib'. (requires matplotlib as dependency)")
    parser.add_argument('-teap', '--calc-top-approx', action='store_true', 
                        help="Calculate the top event using standard approximation.")
    parser.add_argument('-tecut', '--calc-top-cuts', action='store_true',
                        help="Calculate the top event with the alpha-cut method.")
    parser.add_argument('-im', '--calc-importance', action='store_true',
                        help="Calculate the importance measures FIM and FUIM of base events using the alpha cut method.")
    parser.add_argument('-mcs', '--minimum-cut-sets', action='store_true',
                        help="Calculate the minimum cut sets of the fault tree. (for the results)")
    parser.add_argument('-umcs', '--use-minimum-cut-sets', action='store_true', default=False,
                        help="Signal the top event and importance measure calculations to use the minimum cut set representation of the tree.")
    parser.add_argument('-p', '--precision', type=int,
                        help="Sets the precision to the specified number of digits.")
    parser.add_argument('-t', '--time-execution', action='store_true',
                        help="Prints rough estimates of program execution time in seconds to the console (Does not save in file).")
    parser.add_argument('--licence', action='store_true',
                        help="Print out the copiryght notice. NOTE: Using this ignores all other flags.")
    
    # actually take the supplied arguments and parse them
    args = parser.parse_args()

    # GUI call
    if args.gui:
        raise NotImplementedError("GUI is not yet implemented for this software.")
    
    # LICENCE notice
    elif args.licence:
        print(__licence__)
    
    # CLI call
    else:
        # load from supplied file
        fft = FuzzyFaultTree.load_from_file(args.input_file)
        
        if args.precision:
            __PRECISION__ = args.precision
            
        alphaLevels = args.alpha_levels
        if args.number_of_alpha_levels:
            nal = args.number_of_alpha_levels
            assert args.number_of_alpha_levels >= 2, "Defining the number of alpha levels requires atleast a value of 2."
            alphaLevels = [round(al / (nal - 1), __PRECISION__) for al in list(range(nal - 1))]
            alphaLevels.append(1.0)

        results = {}
        if args.time_execution:
            times = {'start': time.time()}
        
        if args.use_minimum_cut_sets:
            fft = fft.calculate_minimum_cut_sets(newTree = True)
            results['minimum-cut-sets'] = {f'MCS-{i+1}': c['inputs'] for i, c in enumerate(fft.logicGates.values()) if i >= 1}
            if args.time_execution: times['mcs'] = time.time()
        elif args.minimum_cut_sets:
            results['minimum-cut-sets'] = {f'MCS-{i+1}': c for i, c in enumerate(fft.calculate_minimum_cut_sets())}
            if args.time_execution: times['mcs'] = time.time()
        
        if args.calc_top_approx:
            results['top-event-standard-approximation'] = fft.calculate_top_event().to_list()
            if args.time_execution: times['teap'] = time.time()
        
        if args.calc_top_cuts:
            results['top-event-alphacuts'] = [fft.calculate_top_event_alphacut(level).to_list() 
                                              for level in alphaLevels]
            if args.time_execution: times['tecut'] = time.time()
        
        if args.calc_importance:
            results['fuzzy-importance-measure'] = fft.calculate_fim(alphaLevels)
            if args.time_execution: times['fim'] = time.time()
            results['fuzzy-uncertainty-importance-measure'] = fft.calculate_fuim(alphaLevels)
            if args.time_execution: times['fuim'] = time.time()

        # output location file or terminal
        if not args.output_file is None:
            if os.path.exists(args.output_file):
                raise Exception(f"Cannot save fuzzy program results to specified file, it already exists / '{args.output_file}'")

            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=4)

        else:
            # prettify printing
            import pprint

            pp.pprint(results)

        if args.time_execution:
            print("------ execution times in seconds ----------")
            timed = {k: round(times[k] - times['start'], __PRECISION__) for k in times.keys() if not (k == 'start')}
            timed['total'] = round(sum(list(timed.values())), __PRECISION__)
            pp.pprint(timed)

        if args.visualize:
            visualize_fft(fft, alphaLevelsTopEvent = alphaLevels, 
                          topEvent = args.calc_top_approx | args.calc_top_cuts, measures = args.calc_importance,
                          )
