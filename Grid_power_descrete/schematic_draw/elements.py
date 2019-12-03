'''
Electrical element symbols for schematic drawing.

Each element is a dictionary with key/values defining how it should be drawn.

Coordinates are all defined in element coordinates, where the element begins
at [0,0] and is drawn from left to right. The drawing engine will then rotate
and translate the element to its final position. A standard resistor is
1 drawing unit long, and with default lead extension will become 3 units long.

Possible dictionary keys:
    name:  A name string for the element. Currently only used for testing.
    paths: A list of each path line in the element. For example, a capacitor
           has two paths, one for each capacitor "plate". On 2-terminal
           elements, the leads will be automatically extended away from the
           first and last points of the first path, and don't need to
           be included in the path.
    base:  Dictionary defining a base element. For example, the variable
           resistor has a base of resistor, then adds an additional path.
    shapes: A list of shape dictionaries.
            'shape' key can be [ 'circle', 'poly', 'arc', 'arrow' ]
            Other keys depend on the shape as follows.
            circle:
                'center': xy center coordinate
                'radius': radius of circle
                'fill'  : [True, False] fill the circle
                'fillcolor' : color for fill
            poly:
                'xy' : List of xy coordinates defining polygon
                'closed': [True, False] Close the polygon
                'fill'  : [True, False] fill the polygon
                'fillcolor' : color for fill
            arc:
                'center' : Center coordinate of arc
                'width', 'height': width and height of arc
                'theta1' : Starting angle (degrees)
                'theta2' : Ending angle (degrees)
                'angle'  : Rotation angle of entire arc
                'arrow'  : ['cw', 'ccw'] Add an arrowhead, clockwise or counterclockwise
            arrow:
                'start'  : start of arrow
                'end'    : end of arrow
                'headwidth', 'headlength': width and length of arrowhead
    theta: Default angle (in degrees) for the element. Overrides the current
           drawing angle.
    anchors: A dictionary defining named positions within the element. For
             example, the NFET element has a 'source', 'gate', and 'drain'
             anchor. Each anchor will become an attribute of the element class
             which can then be used for connecting other elements.

    extend: [True, False] Extend the leads to fill the full element length.
    move_cur: [True, False] Move the drawing cursor location after drawing.
    color: A matplotlib-compatible color for the element. Examples include
           'red', 'blue', '#34ac92'
    drop: Final location to leave drawing cursor.
    lblloc: ['top', 'bot', 'lft', 'rgt'] default location for text label.
            Defaults to 'top'.
    lblofst: Default distance between element and text label.
    labels: List of (label, pos, align) tuples defining text labels to always draw
            in the element. Align is (horiz, vert) tuple of
            (['center', 'left', right'], ['center', 'bottom', 'top'])
    labels: list of label dictionaries. Each label has keys:
            'label' : string label
            'pos'   : xy position
            'align' : (['center', 'left', right'], ['center', 'bottom', 'top']) alignment
            'size'  : font size
'''

import numpy as _np
import re

_gap = [_np.nan, _np.nan]   # To leave a break in the plot

# Resistor is defined as 1 matplotlib plot unit long.
# When default leads are added, the total length will be three units.
_rh = 0.25      # Resistor height
_rw = 1.0 / 6   # Full (inner) length of resistor is 1.0 data unit
_dotr = .075


#Bus
BUS={'name':'BUS','lblofst': .2,'paths':[[[_rh,0], [-_rh,0]]]}

DOT = {
    'name': 'DOT',
    'paths': [[[0, 0]]],
    'shapes': [{'shape': 'circle',
                'center': [0, 0],
                'radius':_dotr,
                'fill': True,
                'fillcolor': 'black'}],
    'theta': 0,
    'extend': False,
    }
ELLIPSIS = {
    'name': 'ELLIPSIS',
    'shapes': [{'shape': 'circle',
                'center': [.5, 0],
                'radius': _dotr/2,
                'fill': True,
                'fillcolor': 'black'},
               {'shape': 'circle',
                'center': [1, 0],
                'radius': _dotr/2,
                'fill': True,
                'fillcolor': 'black'},
               {'shape': 'circle',
                'center': [1.5, 0],
                'radius': _dotr/2,
                'fill': True,
                'fillcolor': 'black'}],
    'extend': False,
    'drop': [2, 0]
    }


GEN = {
    'name': 'Generator',
    'paths': [[ [-0.6, 0], [0, 0], _gap,[-0.6, 0]]],
    'theta': 90.,
    'labels': [{'label': 'G', 'pos': [.45, 0]},{'label':'~', 'pos': [.75, 0]}],
    
    'shapes': [{'shape': 'circle',
                'center': [0.5, 0],
                'radius': 0.5}],
     }

TRANS= {
    'name': 'Transformer',
    'paths': [[[-0.7, 0], [0, 0], _gap, [0.75, 0], [0.95, 0]]],
    'theta': 90.,
    'shapes': [{'shape': 'circle',
                'center': [0.25, 0],
                'radius': 0.25},
                {'shape': 'circle',
                'center': [0.5, 0],
                'radius': 0.25}],
     }

TRANS3w ={
    'name': 'transformet3W',
    'paths': [[[-0.1, 0], [0, 0], _gap, [0.92, 0.25],[1.2,.25],_gap, [0.92, -0.25],[1.2,-.25]]],
    'theta': 90.,
    'shapes': [{'shape': 'circle',
                'center': [0.35, 0],
                'radius': 0.35},
                {'shape': 'circle',
                'center': [0.6, .2],
                'radius': -0.35},
                {'shape': 'circle',
                'center': [0.6, -.2],
                'radius': -0.35}],
    'extend': False,
     }

GRID = {
    'name': 'GRID',
    'paths': [[[0,0],[1,0],[1, _rh*2.5], [_rw*3, _rh*2.5], [_rw*3, -_rh*2.5],
               [1, -_rh*2.5],[1,0], [0, 0]]],
    'extend': False,
    'theta': 0
     }
     
_sw_dot_r = .7
PV_P = {
    'name': 'VPP',
    'paths': [[[0, 0],[0.7,0]]],
    'shapes': [{'shape': 'circle',
                'center': [1.2, 0],
                'radius': 0.5},
             {'shape': 'poly',
                'xy': _np.array([[1.5, -0.3], [0.9, -.3], [1.2, .01]]),
                'fill': False},
              {'shape': 'poly',
                'xy': _np.array([[1.5, 0.3], [0.9, .3], [1.2, .01]]),
                'fill': False}],
    'extend': False,
    }

dis=_rw*6
PV= {
    'name': 'PV',
    'paths': [[[0, 0],[0.8,0], [0.8, _rh], [dis+.8, _rh], [dis+.8, -_rh],
               [0.8, -_rh], [0.8, 0], _gap, [.8, _rh],[1.1, 0],[.8, -_rh]]],
    'extend': False        
    }

_cap_gap = .9
SHUNT = {   
    'name': 'SHUNT',
    'paths': [[[0, 0],[0.7,0], _gap, [0.7, _rh*1.5], [0.7, -_rh*1.5], _gap,
               [_cap_gap, _rh*1.5], [_cap_gap, -_rh*1.5], _gap,
               [_cap_gap, 0],[1.1,0],[1.1, _rh*1.2], [1.1, -_rh*1.2]]],
     'anchors': {'center': [_rh, 0]},
     'extend': False
    }

LINE = {'name': 'LINE', 'paths': [_np.array([[0, 0]])]}


LOAD = {
    'name': 'GND_SIG',
    'paths': [[[0, 0], [.7, 0]]],
    'shapes':[{'shape': 'poly',
                'xy': _np.array([[0.7, 0.3], [0.7, -.3], [1.2, 0]]),
                'fill': True}],
    'move_cur': False,
    'extend': False,
    'theta': 0
    }
_sw_dot_r = .12
SWITCH = {
    'name': 'SWITCH',
    'paths': [[[0, 0],[.7, 0], _gap, [0.82, .1], [1.5, .45], _gap, [1.67, 0],[2,0]]],
    'shapes': [{'shape': 'circle',
                'center': [1.5, 0],
                'radius': _sw_dot_r},
               {'shape': 'circle',
                'center': [1-_sw_dot_r, 0],
                'radius': _sw_dot_r}],
    'extend': False
    }

SWITCH_CLOSE = {
    'name': 'SWITCH_SPST_OPEN',
    'base': SWITCH,
    'shapes': [{'shape': 'arc',
                'center': [1, .09],
                'width': .5,
                'height': .7,
                'theta1': -10,
                'theta2': 70,
                'arrow': 'ccw'}],
    'extend': False,
    }

SWITCH_OPEN = {
    'name': 'SWITCH_SPST_CLOSE',
    'base': SWITCH,
    'shapes': [{'shape': 'arc',
                'center': [1, .25],
                'width': .5,
                'height': .75,
                'theta1': -10,
                'theta2': 70,
                'arrow': 'cw'}],
    'extend': False,
    }

dis=_rw*2
IMPEDANCE = {
    'name': 'IMPEDANCE',
    'paths': [[[0, 0],[0.8,0], [0.8, _rh], [dis+.8, _rh], [dis+.8, -_rh],
               [0.8, -_rh], [0.8, 0], _gap, [.8+dis, 0],[1.6+dis, 0]]],
    'labels': [{'label': 'Z', 'pos': [.95, 0]}] ,
    'extend': False        
    }

DIODE = {
    'name': 'DIODE',
    'paths': [[[0, 0], _gap, [_rh*1.4, _rh], [_rh*1.4, -_rh], _gap, [_rh*1.4, 0]]],
    'anchors': {'center': [_rh, 0]},
    'shapes': [{'shape': 'poly',
                'xy': _np.array([[0, _rh], [_rh*1.4, 0], [0, -_rh]]),
                'fill': True}],
    'extend': False,
    }

LED = {
    'name': 'LED',
    'base': DIODE,
    'paths': [[[_rh, _rh*1.5], [_rh*2, _rh*3.25]]],  # Duplicate arrow with a path to work around matplotlib autoscale bug.
    'shapes': [{'shape': 'arrow',
                'start': [_rh, _rh*1.5],
                'end': [_rh*2, _rh*3.25],
                'headwidth': .12,
                'headlength': .2},
               {'shape': 'arrow',
                'start': [_rh*.1, _rh*1.5],
                'end': [_rh*1.1, _rh*3.25],
                'headwidth': .12,
                'headlength': .2}]
    }
SOURCE = {
    'name': 'SOURCE',
    'paths': [[[0, 0], [0, 0], _gap, [1, 0], [1, 0]]],
    'theta': 90.,
    'shapes': [{'shape': 'circle',
                'center': [0.5, 0],
                'radius': 0.5}],
     }

_plus_len = .2
SOURCE_V = {
    'name': 'SOURCE_V',
    'base': SOURCE,
    'paths': [[[.25, -_plus_len/2], [.25, _plus_len/2]],     # '-' sign
              [[.75-_plus_len/2, 0], [.75+_plus_len/2, 0]],  # '+' sign
              [[.75, -_plus_len/2], [.75, _plus_len/2]],    # '+' sign
              ]
    }

SOURCE_I = {
    'name': 'SOURCE_I',
    'base': SOURCE,
    'shapes': [{'shape': 'arrow',
                'start': [.75, 0],
                'end': [.25, 0]}]
    }

# Independent sources
SOURCE = {
    'name': 'SOURCE',
    'paths': [[[0, 0], [0, 0], _gap, [1, 0], [1, 0]]],
    'theta': 90.,
    'shapes': [{'shape': 'circle',
                'center': [0.5, 0],
                'radius': 0.5}],
     }
# Meters
METER_V = {
    'name': 'METER_V',
    'base': SOURCE,
    'labels': [{'label': 'V', 'pos': [.5, 0]}]
    }

METER_I = {
    'name': 'METER_I',
    'base': SOURCE,
    'labels': [{'label': 'I', 'pos': [.5, 0]}]
    }

METER_OHM = {
    'name': 'METER_OHM',
    'base': SOURCE,
    'labels': [{'label': '$\Omega$', 'pos': [.5, 0]}]
    }
ARROWLINE = {
    'name': 'ARROWLINE',
    'paths': [[[0, 0], [1, 0]]],
    'shapes': [{'shape': 'arrow',
                'start': [0, 0],
                'end': [0.55, 0],
                'headwidth': .2,
                'headlength': .3}],
    'lblofst': .2,
    }
_a = .25
_b = .7
_t = _np.linspace(1.4, 3.6*_np.pi, 100)
_x = _a*_t - _b*_np.sin(_t)
_y = _a - _b * _np.cos(_t)
_x = (_x - _x[0])  # Scale to about the right size
_x = _x / _x[-1]
_y = (_y - _y[0]) * .25
_lamp_path = _np.transpose(_np.vstack((_x, _y)))
LAMP = {
    'name': 'LAMP',
    'base': SOURCE,
    'paths': [_lamp_path]
    }
H_PV={
    'name':'house_PV',
    'paths':[[[0,0]]],
    'shapes': [{'shape': 'circle',
                'center': [1.1,0],
                'radius': 0.61},
                {'shape': 'poly',
                'xy': _np.array([[1.5, -0.3], [0.9, -.3], [1.2, .01]]),
                'fill': False},
                {'shape': 'poly',
                'xy': _np.array([[1.5, 0.3], [0.9, .3], [1.2, .01]]),
                'fill': False}]
}
#House
HOUSEPB = {
    'name': 'House',
    'paths':[[[0,0],[.1,0],[.1,-.7] ,[.4, -.7],[.3,-.9],[.1-.38,-.9],[.1-.3,-.7],[.1,-.7],_gap,
    [.4, -.7],[.52,-.86],[.25,-.9],[.25,-1.4],[.1-.3,-1.4],[.1-.3,-.9],_gap,[.45,-.88],[.45,-1.3],[.25,-1.4],
    _gap,[.58,-1.01],[.59,-1.02],[.59,-1.08],[.8,-1.08],[.8,-1.01],[.81,-1.02],[.81,-1.3],[.58,-1.3],[.58,-1.01]]],
    'shapes': [{'label':'PV',
                'shape': 'circle',
                'center': [.71, -.8],
                'radius': 0.13,         
                'fill':True,
                'fillcolor': 'red'}],
    'labels': [{'label': 'pv', 'pos': [1.05, -.8]},{'label': 'Bat', 'pos': [1.05, -1.2]}],
    'extend': False,
    }
HOUSEP = {
    'name': 'House',
    'paths':[[[0,0],[.1,0],[.1,-.7] ,[.4, -.7],[.3,-.9],[.1-.38,-.9],[.1-.3,-.7],[.1,-.7],_gap,
    [.4, -.7],[.52,-.86],[.25,-.9],[.25,-1.4],[.1-.3,-1.4],[.1-.3,-.9],_gap,[.45,-.88],[.45,-1.3],[.25,-1.4]]],
    'shapes': [{'label':'PV',
                'shape': 'circle',
                'center': [.71, -.8],
                'radius': 0.13,         
                'fill':True,
                'fillcolor': 'red'}],
    'labels': [{'label': 'pv', 'pos': [1.05, -.8]}],
    'extend': False,
    }
HOUSE = {
    'name': 'House',
    'paths':[[[0,0],[0,-0.7], [.3, -.7],[.2,-.9],[-.38,-.9],[-.3,-.7],[0,-.7],_gap,
    [.3, -.7],[.42,-.86],[.15,-.9],[.15,-1.4],[-.3,-1.4],[-.3,-.9],_gap,[0.35,-.88],[.35,-1.3],[.15,-1.4]]],
    'extend': False,
    }