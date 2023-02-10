import numpy as np

import KratosMultiphysics
import KratosMultiphysics.RomApplication as KratosROM

from rom_analysis import CreateRomAnalysisInstance
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis


"""
For user-scripting it is intended that a new class is derived
from StructuralMechanicsAnalysis to do modifications
"""

if __name__ == "__main__":

    with open("ProjectParameters_rom.json",'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    analysis_stage_class = StructuralMechanicsAnalysis

    model = KratosMultiphysics.Model()
    simulation = CreateRomAnalysisInstance(analysis_stage_class, model, parameters)
    simulation.Run()
