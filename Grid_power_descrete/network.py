import six
import copy
from numpy import nan, isnan, arange, dtype, zeros
import numpy as np
import pandas as pd
from config import *
from standard_lib import StdLib
from collections import MutableMapping

def _preserve_dtypes(df, dtypes):
    for item, dtype in list(dtypes.iteritems()):
        if df.dtypes.at[item] != dtype:
            try:
                df[item] = df[item].astype(dtype)
            except ValueError:
                df[item] = df[item].astype(float)

def get_free_id(df):
    """
    Returns next free ID in a dataframe
    """
    return np.int64(0) if len(df) == 0 else df.index.values.max() + 1

class ADict(dict, MutableMapping):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # to prevent overwrite of internal attributes by new keys
        # see _valid_name()
        self._setattr('_allow_invalid_attributes', False)

    def _build(self, obj, **kwargs):
        """
        We only want dict like elements to be treated as recursive AttrDicts.
        """
        return obj

    # --- taken from AttrDict

    def __getstate__(self):
        return self.copy(), self._allow_invalid_attributes

    def __setstate__(self, state):
        mapping, allow_invalid_attributes = state
        self.update(mapping)
        self._setattr('_allow_invalid_attributes', allow_invalid_attributes)

    @classmethod
    def _constructor(cls, mapping):
        return cls(mapping)

    # --- taken from MutableAttr

    def _setattr(self, key, value):
        """
        Add an attribute to the object, without attempting to add it as
        a key to the mapping (i.e. internals)
        """
        super(MutableMapping, self).__setattr__(key, value)

    def __setattr__(self, key, value):
        """
        Add an attribute.
        key: The name of the attribute
        value: The attributes contents
        """
        if self._valid_name(key):
            self[key] = value
        elif getattr(self, '_allow_invalid_attributes', True):
            super(MutableMapping, self).__setattr__(key, value)
        else:
            raise TypeError(
                "'{cls}' does not allow attribute creation.".format(
                    cls=self.__class__.__name__
                )
            )

    def _delattr(self, key):
        """
        Delete an attribute from the object, without attempting to
        remove it from the mapping (i.e. internals)
        """
        super(MutableMapping, self).__delattr__(key)

    def __delattr__(self, key, force=False):
        """
        Delete an attribute.
        key: The name of the attribute
        """
        if self._valid_name(key):
            del self[key]
        elif getattr(self, '_allow_invalid_attributes', True):
            super(MutableMapping, self).__delattr__(key)
        else:
            raise TypeError(
                "'{cls}' does not allow attribute deletion.".format(
                    cls=self.__class__.__name__
                )
            )

    def __call__(self, key):
        """
        Dynamically access a key-value pair.
        key: A key associated with a value in the mapping.
        This differs from __getitem__, because it returns a new instance
        of an Attr (if the value is a Mapping object).
        """
        if key not in self:
            raise AttributeError(
                "'{cls} instance has no attribute '{name}'".format(
                    cls=self.__class__.__name__, name=key
                )
            )

        return self._build(self[key])

    def __getattr__(self, key):
        """
        Access an item as an attribute.
        """
        if key not in self or not self._valid_name(key):
            raise AttributeError(
                "'{cls}' instance has no attribute '{name}'".format(
                    cls=self.__class__.__name__, name=key
                )
            )

        return self._build(self[key])

    @classmethod
    def _valid_name(cls, key):
        """
        Check whether a key is a valid attribute name.
        A key may be used as an attribute if:
         * It is a string
         * The key doesn't overlap with any class attributes (for Attr,
            those would be 'get', 'items', 'keys', 'values', 'mro', and
            'register').
        """
        return (
                isinstance(key, six.string_types) and
                not hasattr(cls, key)
        )

class Gridnet(ADict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(args[0], self.__class__):
            net = args[0]
            self.clear()
            self.update(**net.deepcopy())

    def deepcopy(self):
        return copy.deepcopy(self)

    def __repr__(self):  # pragma: no cover
        r = "This Gridapower network includes the following parameter tables:"
        par = []
        res = []
        for tb in list(self.keys()):
            if not tb.startswith("_") and isinstance(self[tb], pd.DataFrame) and len(self[tb]) > 0:
                if 'res_' in tb:
                    res.append(tb)
                else:
                    par.append(tb)
        for tb in par:
            length = len(self[tb])
            r += "\n   - %s (%s %s)" % (tb, length, "elements" if length > 1 else "element")
        if res:
            r += "\n and the following results tables:"
            for tb in res:
                length = len(self[tb])
                r += "\n   - %s (%s %s)" % (tb, length, "elements" if length > 1 else "element")
        return r


class Network(object):
    """
    main class for total power network
    call data frame form config data and pass data to powerNet which helps to make the dataframe
    """
    net=Gridnet(Net)
    component=[]
    for s in net:
        component.append(s)
        if isinstance(net[s], list):
            net[s] = pd.DataFrame(zeros(0, dtype=net[s]), index=pd.Int64Index([]))

    def __init__(self,new=True):
        if new:
            self.net.clear()
            self.net=Gridnet(Net)
            for s in self.net:
                if isinstance(self.net[s], list):
                    self.net[s] = pd.DataFrame(zeros(0, dtype=self.net[s]), index=pd.Int64Index([]))

        self.storagetypes=storagetypes
        self.pvtypes=pvtypes
        self.bus        =self.net.bus
        self.line       =self.net.line
        self.load       =self.net.load
        self.house       =self.net.house
        self.housepv       =self.net.housepv
        self.housepvb       =self.net.housepvb  
        self.ext_grid   =self.net.ext_grid
        self.switch     =self.net.switch
        self.gen        = self.net.gen
        self.pv       = self.net.pv
        self.storage    = self.net.storage
        self.bus_count  =0
        self.line_count =[]
        self.plot_data  =[]


    def create_bus(self, name=None, index=None, geodata=None, type="b",
               zone=None, in_service=True,  coords=None, **kwargs):

        if index is not None and index in self.net["bus"].index:
            raise UserWarning("A bus with index %s already exists" % index)

        if index is None:
            index = get_free_id(self.net["bus"])
        
        if name==None:
            name='bus'+str(index)

        self.bus_count+=1
        dtypes = self.net.bus.dtypes

        self.net.bus.loc[index, ["name", "type", "zone", "in_service"]] = \
            [name, type, zone, bool(in_service)]

        # and preserve dtypes
        _preserve_dtypes(self.net.bus, dtypes)

        if geodata is not None:
            if len(geodata) != 2:
                raise UserWarning("geodata must be given as (x, y) tupel")
            self.net["bus_geodata"].loc[index, ["x", "y"]] = geodata

        if coords is not None:
            self.net["bus_geodata"].loc[index, "coords"] = coords

        return index


    def create_line(self,from_bus, to_bus, length_km, name=None, index=None, geodata=None,
                 parallel=1, in_service=True, direct="right",theta=0):

        for b in [from_bus, to_bus]:
            if b not in self.net["bus"].index.values:
                raise UserWarning("Line %s tries to attach to non-existing bus %s" % (name, b))
        
        if index is None:
            index = get_free_id(self.net["line"])

        if index in self.net["line"].index:
            raise UserWarning("A line with index %s already exists" % index)
        
        if name==None:
            name='line'+str(index)

        self.plot_data.append(["line", ("from_bus", from_bus),("to_bus", to_bus),(direct,theta)])
        self.line_count.append([("from_bus", from_bus),("to_bus", to_bus)])
        # store dtypes
        dtypes = self.net.line.dtypes

        self.net.line.loc[index, ["name", "from_bus","to_bus", "length_km", "in_service"]] = \
            [name, from_bus, to_bus,length_km, bool(in_service)]

        # and preserve dtypes
        _preserve_dtypes(self.net.line, dtypes)

        if geodata is not None:
            self.net["line_geodata"].loc[index, "coords"] = geodata

        return index


    def create_load(self, bus, p_w,name=None, scaling=1., index=None,
                in_service=True, type=None,controllable=nan,direct="down",option=" "):

        if bus not in self.net["bus"].index.values:
            raise UserWarning("Cannot attach to bus %s, bus does not exist" % bus)

        if index is None:
            index = get_free_id(self.net["load"])

        if index in self.net["load"].index:
            raise UserWarning("A load with the id %s already exists" % index)
        
        if name==None:
            name='load'+str(index)
        
        self.plot_data.append(["load", ("bus", bus),(direct,option)])        
        # store dtypes
        dtypes = self.net.load.dtypes

        self.net.load.loc[index, ["name", "bus", "p_w", "scaling","in_service"]] = \
            [name, bus, p_w, scaling,bool(in_service)]

        # and preserve dtypes
        _preserve_dtypes(self.net.load, dtypes)

        if not isnan(controllable):
            if "controllable" not in self.net.load.columns:
                self.net.load.loc[:, "controllable"] = pd.Series()

            self.net.load.loc[index, "controllable"] = bool(controllable)
        else:
            if "controllable" in self.net.load.columns:
                self.net.load.loc[index, "controllable"] = False
        return index,

    def create_house(self, bus, p_w, pv=False, storage=False, scaling=1., index=None,
                in_service=True, type=None,controllable=nan, direct="right",option=""):
        
        if pv:
            raise UserWarning("use create_housepv instead of create_house, pv not exist")
        if storage and pv:
            raise UserWarning("use create_housepvb instead of create_house, pv not exist")
        if storage:
            raise UserWarning("house with battery is not exist")

        if bus not in self.net["bus"].index.values:
            raise UserWarning("Cannot attach to bus %s, bus does not exist" % bus)
        #self.plot_data.append(["load", ("bus", bus),(direct,option)])
        if index is None:
            index = get_free_id(self.net["house"])
    
        if index in self.net["house"].index:
            raise UserWarning("A load with the id %s already exists" % index)
        name='house'+str(index)

        load=self.create_load(bus,p_w,name=name,direct="right",option="house")
        # store dtypes
        dtypes = self.net.house.dtypes

        self.net.house.loc[index, ["name", "bus", "p_w", "pv_w", "st_w", "scaling","in_service",]] = \
            [name, bus, p_w, pv, storage, scaling, bool(in_service)]

        # and preserve dtypes
        _preserve_dtypes(self.net.house, dtypes)
        
        if not isnan(controllable):
            if "controllable" not in self.net.house.columns:
                self.net.house.loc[:, "controllable"] = pd.Series()

            self.net.house.loc[index, "controllable"] = bool(controllable)
        else:
            if "controllable" in self.net.house.columns:
                self.net.house.loc[index, "controllable"] = False

        return index

    def create_housepv(self, bus, p_w, pv, storage=False,scaling=1., index=None,
                in_service=True, type=None,controllable=nan, direct="right",option=""):

        if storage:
            raise UserWarning("use create_housepvb instead of create_housepv, pv not exist")

        if bus not in self.net["bus"].index.values:
            raise UserWarning("Cannot attach to bus %s, bus does not exist" % bus)
        #self.plot_data.append(["load", ("bus", bus),(direct,option)])
        if index is None:
            index = get_free_id(self.net["housepv"])
    
        if index in self.net["housepv"].index:
            raise UserWarning("A load with the id %s already exists" % index)

        assert pv in list(self.pvtypes.keys()),"pv std_lib is not exit"

        max_p_w, min_p_w=self.pvtypes[pv].values()
        name='housepv'+str(index)
        pv=self.create_pv(bus,p_w=max_p_w,name=name)
        load=self.create_load(bus,p_w,name=name,direct="right",option="housepv")
        
        # store dtypes
        dtypes = self.net.housepv.dtypes

        self.net.housepv.loc[index, ["name", "bus", "p_w", "pv", "st_w", "scaling","in_service",]] = \
            [name, bus, p_w, pv, storage, scaling, bool(in_service)]

        # and preserve dtypes
        _preserve_dtypes(self.net.housepv, dtypes)
        
        if not isnan(controllable):
            if "controllable" not in self.net.housepv.columns:
                self.net.housepv.loc[:, "controllable"] = pd.Series()

            self.net.housepv.loc[index, "controllable"] = bool(controllable)
        else:
            if "controllable" in self.net.load.columns:
                self.net.housepv.loc[index, "controllable"] = False
        return index, name

    def create_housepvb(self, bus, p_w, pv, storage, scaling=1., index=None,
                in_service=True, type=None,controllable=nan, direct="right",option=""):

        if bus not in self.net["bus"].index.values:
            raise UserWarning("Cannot attach to bus %s, bus does not exist" % bus)
        #self.plot_data.append(["load", ("bus", bus),(direct,option)])
        if index is None:
            index = get_free_id(self.net["housepvb"])
    
        if index in self.net["housepvb"].index:
            raise UserWarning("A load with the id %s already exists" % index)

        assert pv in list(self.pvtypes.keys()),"pv std_lib is not exit"
        assert storage in list(self.storagetypes.keys()),"battery std_lib is not exit"

        mv_p_w, mp_p_w=self.pvtypes[pv].values()
        mt_p_wm,mt_p_wl,max_discharge,max_charge=self.storagetypes[storage].values()
        name='housepvb'+str(index)

        pv=self.create_pv(bus,p_w,name=name)
        load=self.create_load(bus,p_w,name=name,direct="right",option="housepvb")
        storage=self.create_storage(bus,max_p_w=mt_p_wm,start_p_w=mt_p_wm,minimum_p_w=mt_p_wl,name=name)
        # store dtypes
        dtypes = self.net.housepvb.dtypes

        self.net.housepvb.loc[index, ["name", "bus", "p_w", "pv", "st_w", "scaling","in_service"]] = \
            [name, bus, p_w, pv, storage, scaling, bool(in_service)]

        # and preserve dtypes
        _preserve_dtypes(self.net.housepvb, dtypes)
        
        if not isnan(controllable):
            if "controllable" not in self.net.housepvb.columns:
                self.net.housepv.loc[:, "controllable"] = pd.Series()

            self.net.housepvb.loc[index, "controllable"] = bool(controllable)
        else:
            if "controllable" in self.net.load.columns:
                self.net.housepvb.loc[index, "controllable"] = False

        return index, name


    def create_ext_grid(self, bus, max_p_mw, name="Grid", in_service=True,index=None, **kwargs):

        if bus not in self.net["bus"].index.values:
            raise UserWarning("Cannot attach to bus %s, bus does not exist" % bus)

        if index is not None and index in self.net["ext_grid"].index:
            raise UserWarning("An external grid with with index %s already exists" % index)

        if index is None:
            index = get_free_id(self.net["ext_grid"])
        self.plot_data.append(["ext_grid", ("bus", bus)])

        # store dtypes
        dtypes = self.net.ext_grid.dtypes

        self.net.ext_grid.loc[index, ["bus", "name", "max_p_mw","in_service"]] = \
            [bus, name, max_p_mw, bool(in_service)]

        if not isnan(max_p_mw):
            if "max_p_mw" not in self.net.ext_grid.columns:
                self.net.ext_grid.loc[:, "max_p_mw"] = pd.Series()

            self.net.ext_grid.loc[index, "max_p_mw"] = float(max_p_mw)

            # and preserve dtypes
        _preserve_dtypes(self.net.ext_grid, dtypes)
        return index

    def create_transformer(self,from_bus, to_bus,name=None, index=None, geodata=None,
                df=1., parallel=1, in_service=True, direct="right",theta=0):
        for b in [from_bus, to_bus]:
            if b not in self.net["bus"].index.values:
                raise UserWarning("transformer %s tries to attach to non-existing bus %s" % (name,b))
        
        if index is None:
            index = get_free_id(self.net["transformer"])

        if index in self.net["transformer"].index:
            raise UserWarning("A transformer with index %s already exists" % index)

        self.plot_data.append(["transformer", ("hv_bus", hv_bus),("lv_bus",lv_bus)])
        # store dtypes
        dtypes = self.net.transformer.dtypes

        self.net.transformer.loc[index, ["name", "from_bus","to_bus", "zone", "in_service"]] = \
            [name, from_bus, to_bus, zone, bool(in_service)]

        # and preserve dtypes
        _preserve_dtypes(self.net.transformer, dtypes)

        if geodata is not None:
            self.net["transformer_geodata"].loc[index, "coords"] = geodata

        return index


    def create_switch(self, from_bus, to_bus,closed=True, type=None, name=None, index=None,direct="down"):
        
        for b in [from_bus, to_bus]:
            if b not in self.net["bus"].index.values:
                raise UserWarning("switch %s tries to attach to non-existing bus %s" % (name,b))
        
        if index is None:
            index = get_free_id(self.net["switch"])

        if index in self.net["switch"].index:
            raise UserWarning("A transformer with index %s already exists" % index)

        # store dtypes
        dtypes = self.net.switch.dtypes

        self.switch.loc[index, ["bus", "from_bus", "to_bus", "closed", "type", "name"]] = \
            [bus, from_bus, to_bus, closed, type, name]

        # and preserve dtypes
        _preserve_dtypes(self.net.switch, dtypes)

        return index

    def create_gen(self, bus, p_w, name=None, index=None, scaling=1., type=None, 
               controllable=nan, in_service=True,direct="down"):

        if bus not in self.net["bus"].index.values:
            raise UserWarning("Cannot attach to bus %s, bus does not exist" % bus)
        if index is None:
            index = get_free_id(self.net["gen"])

        if index in self.net["gen"].index:
            raise UserWarning("A generator with the id %s already exists" % index)
        # store dtypes
        dtypes = self.net.gen.dtypes

        self.net.gen.loc[index, ["name", "bus", "p_w","type","in_service","scaling"]] = \
            [name, bus, p_w, type, bool(in_service), scaling]

        # and preserve dtypes
        _preserve_dtypes(self.net.gen, dtypes)

        return index

    def create_pv(self, bus, p_w, name=None, index=None, scaling=1., type=None, 
               controllable=nan, in_service=True,direct="down"):

        if bus not in self.net["bus"].index.values:
            raise UserWarning("Cannot attach to bus %s, bus does not exist" % bus)
        if index is None:
            index = get_free_id(self.net["pv"])

        if index in self.net["pv"].index:
            raise UserWarning("A generator with the id %s already exists" % index)

        if name==None:
            name='pv'+str(index)
        # store dtypes
        dtypes = self.net.pv.dtypes

        self.net.pv.loc[index, ["name", "bus", "p_w","type","in_service","scaling"]] = \
            [name, bus, p_w, type, bool(in_service), scaling]

        # and preserve dtypes
        _preserve_dtypes(self.net.pv, dtypes)

        return index
    
    def create_storage(self, bus, max_p_w, start_p_w, minimum_p_w, name=None, index=None, scaling=1., type=None, 
                        in_service=True, controllable = nan,direct="down"):

        if bus not in self.net["bus"].index.values:
            raise UserWarning("Cannot attach to bus %s, bus does not exist" % bus)

        if index is None:
            index = get_free_id(self.net["storage"])

        if index in self.net["storage"].index:
            raise UserWarning("A storage with the id %s already exists" % index)

        # store dtypes
        dtypes = self.net.storage.dtypes

        self.net.storage.loc[index, ["name", "bus", "max_p_w", "start_SOC", "minimum_p_w","end_SOC", "scaling", "in_service"]] = \
            [name, bus, max_p_w, start_p_w, minimum_p_w,nan,scaling, bool(in_service)]

        # and preserve dtypes
        _preserve_dtypes(self.net.storage, dtypes)
        return index

    def create_pwl_cost(self, element, et, points, power_type="p", index=None):
        if index is None:
            index = get_free_id(self.net["pwl_cost"])

        if index in self.net["pwl_cost"].index:
            raise UserWarning("A piecewise_linear_cost with the id %s already exists" % index)

        self.net.pwl_cost.loc[index, ["power_type", "element", "et"]] = \
            [power_type, element, et]
        self.net.pwl_cost.points.loc[index] = points
        return index

    def create_poly_cost(self, element, et, cp1_eur_per_mw, cp0_eur=0, cq1_eur_per_mvar=0,
                           cq0_eur=0, cp2_eur_per_mw2=0, cq2_eur_per_mvar2=0, type="p", index=None):

        if index is None:
            index = get_free_id(self.net["poly_cost"])
        columns = ["element", "et", "cp0_eur", "cp1_eur_per_mw", "cq0_eur", "cq1_eur_per_mvar",
                "cp2_eur_per_mw2", "cq2_eur_per_mvar2"]
        variables = [element, et, cp0_eur, cp1_eur_per_mw, cq0_eur, cq1_eur_per_mvar,
                    cp2_eur_per_mw2, cq2_eur_per_mvar2]
        self.net.poly_cost.loc[index, columns] = variables
        self.net["mode"]="opf"
        return index

    def all_element(self):
        for key in self.__dict__:
            print(key)

    def set_result(self):
        self.res_bus    = self.net.res_bus
        self.res_line   =self.net.res_line
        self.res_cost   =self.net.res_cost
        self.res_ext_grid=self.net.res_ext_grid
        self.res_load   =self.net.res_load
        self.res_pv   = self.net.res_pv
        self.res_storage=self.net.res_storage
        self.res_shunt  =self.net.res_shunt
        self.res_gen    = self.net.res_gen
        self.res_ward   = self.net.res_ward
        self.res_xward  = self.net.res_xward
        self.res_dcline = self.net.res_dcline
