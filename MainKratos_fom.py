import KratosMultiphysics

from fom_analysis import CreateRomAnalysisInstance
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis

if __name__ == "__main__":

    with open("ProjectParameters_fom.json", 'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    analysis_stage_class = StructuralMechanicsAnalysis

    global_model = KratosMultiphysics.Model()
    simulation = CreateRomAnalysisInstance(analysis_stage_class, global_model, parameters)
    
    simulation.Run()

