import pandas as pd
import warnings
from config import storagetypes,pvtypes

import logging
logger = logging.getLogger(__name__)

class StdLib(object):

    def __init__(self,net):
        self.net=net
        self.storagetypes=storagetypes
        self.pvtypes=pvtypes

    def create_std_type(self, data, name, element, overwrite=True, check_required=True):

        if type(data) != dict:
            raise UserWarning("type data has to be given as a dictionary of parameters")

        if check_required:
            if element == "storage":
                required = ["max_p_w", "minimum_p_w", "max_discharge", "max_charge"]
            elif element == "pv":
                required = ["max_p_w","min_p_w"]
            else:
                raise ValueError("Unkown element type %s" % element)
            for par in required:
                if par not in data:
                    raise UserWarning("%s is required as %s type parameter" % (par, element))
        library = self.net.std_types[element]
        if overwrite or not (name in library):
            library.update({name: data})


    def create_std_types(self, data, element, overwrite=True, check_required=True):
      
        for name, typdata in data.items():
            self.create_std_type( data=typdata, name=name, element=element, overwrite=overwrite,
                            check_required=check_required)
                            
    def send_std_data(self,element,std_name):
        library = self.net.std_types[element]
        if name in library:
            return library[name]
        else:
            raise UserWarning("Unknown standard %s type %s" % (element, name))


    def load_std_type(self, name, element):
        """
        Loads standard type data from the linetypes data base. Issues a warning if
        linetype is unknown.

        INPUT:
            **net** - The pandapower network

            **name** - name of the standard type as string

            **element** - "line", "trafo" or "trafo3w"

        OUTPUT:
            **typedata** - dictionary containing type data
        """
        library = self.net.std_types[element]
        if name in library:
            return library[name]
        else:
            raise UserWarning("Unknown standard %s type %s" % (element, name))


    def std_type_exists(self, name, element):
        """
        Checks if a standard type exists.

        INPUT:
            **net** - pandapower Network

            **name** - name of the standard type as string

            **element** - type of element ("line" or "trafo")

        OUTPUT:
            **exists** - True if standard type exists, False otherwise
        """
        library = self.net.std_types[element]
        return name in library


    def delete_std_type(self, name, element):
        """
        Deletes standard type parameters from database.

        INPUT:
            **net** - pandapower Network

            **name** - name of the standard type as string

            **element** - type of element ("line" or "trafo")

        """
        library = self.net.std_types[element]
        if name in library:
            del library[name]
        else:
            raise UserWarning("Unknown standard %s type %s" % (element, name))


    def available_std_types(self, element):
        """
        Returns all standard types available for this network as a table.

        INPUT:
            **net** - pandapower Network

            **element** - type of element ("line" or "trafo")

        OUTPUT:
            **typedata** - table of standard type parameters

        """
        std_types = pd.DataFrame(self.net.std_types[element]).T
        try:
            return std_types.infer_objects()
        except AttributeError:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return std_types.convert_objects()

    def add_basic_std_types(self):
        if "std_types" not in self.net:
            self.net.std_types = {"storage": {}, "pv": {}}
        self.create_std_types(data=self.pvtypes, element="pv")
        self.create_std_types(data=self.storagetypes, element="storage")
        return self.storagetypes, self.pvtypes
        
    