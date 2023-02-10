import KratosMultiphysics
import KratosMultiphysics.RomApplication as KratosROM

from nnm_analysis import CreateRomAnalysisInstance
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis

if __name__ == "__main__":

    with open("ProjectParameters_nnm.json", 'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    # analysis_stage_module_name = parameters["analysis_stage"].GetString()
    # analysis_stage_class_name  = analysis_stage_module_name.split('.')[-1]
    # analysis_stage_class_name  = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

    analysis_stage_class = StructuralMechanicsAnalysis

    global_model = KratosMultiphysics.Model()
    simulation = CreateRomAnalysisInstance(analysis_stage_class, global_model, parameters)
    
    simulation.CreateModel()
    simulation.Run()
