# Call and run the class

from Ni_Pt_NH3 import Wei_NH3_model

x = Wei_NH3_model()
x.build_template()
x.generate_defected()
x.template_to_KMC_lattice()
x.show_all()