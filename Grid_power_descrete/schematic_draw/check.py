from schemdraw import Drawing
import elements as e
d = Drawing(unit=1,fontsize=9,lw=2)
d.add(e.HOUSE)
d.draw()