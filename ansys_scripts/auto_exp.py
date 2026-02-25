Model = ExtAPI.DataModel.Project.Model
Geometry = Model.Geometry
Material = Model.Materials
CS = Model.CoordinateSystems
Mesh = Model.Mesh
Analysis = Model.Analyses[0]
Static_Sol = Analysis.Solution


string = 'num_point,force,p1,p2,p3,p4,p5,p6,p7\n'
for i in range(1):
    for f in range(1):
        Load = Analysis.AddForce()
        Load.Location = ExtAPI.DataModel.GetObjectsByName("in" + str(i+1))[0]
        Load.DefineBy = LoadDefineBy.Components
        Load.XComponent.Output.SetDiscreteValue(0, Quantity(-5*(f+3), "N"))
        
        USUM1 = Static_Sol.AddUserDefinedResult()
        USUM1.ScopingMethod = GeometryDefineByType.Geometry
        USUM1.Location = ExtAPI.DataModel.GetObjectsByName("p1")[0]
        USUM1.Expression = "uy"
        USUM1.OutputUnit = UnitCategoryType.Stress
        
        USUM2 = Static_Sol.AddUserDefinedResult()
        USUM2.ScopingMethod = GeometryDefineByType.Geometry
        USUM2.Location = ExtAPI.DataModel.GetObjectsByName("p2")[0]
        USUM2.Expression = "uy"
        USUM2.OutputUnit = UnitCategoryType.Stress
        
        USUM3 = Static_Sol.AddUserDefinedResult()
        USUM3.ScopingMethod = GeometryDefineByType.Geometry
        USUM3.Location = ExtAPI.DataModel.GetObjectsByName("p3")[0]
        USUM3.Expression = "uy"
        USUM3.OutputUnit = UnitCategoryType.Stress

        USUM4 = Static_Sol.AddUserDefinedResult()
        USUM4.ScopingMethod = GeometryDefineByType.Geometry
        USUM4.Location = ExtAPI.DataModel.GetObjectsByName("p4")[0]
        USUM4.Expression = "uy"
        USUM4.OutputUnit = UnitCategoryType.Stress
        
        USUM5 = Static_Sol.AddUserDefinedResult()
        USUM5.ScopingMethod = GeometryDefineByType.Geometry
        USUM5.Location = ExtAPI.DataModel.GetObjectsByName("p5")[0]
        USUM5.Expression = "uy"
        USUM5.OutputUnit = UnitCategoryType.Stress
        
        USUM6 = Static_Sol.AddUserDefinedResult()
        USUM6.ScopingMethod = GeometryDefineByType.Geometry
        USUM6.Location = ExtAPI.DataModel.GetObjectsByName("p6")[0]
        USUM6.Expression = "uy"
        USUM6.OutputUnit = UnitCategoryType.Stress

        USUM7 = Static_Sol.AddUserDefinedResult()
        USUM7.ScopingMethod = GeometryDefineByType.Geometry
        USUM7.Location = ExtAPI.DataModel.GetObjectsByName("p7")[0]
        USUM7.Expression = "uy"
        USUM7.OutputUnit = UnitCategoryType.Stress
        
        Static_Sol.Solve(True)

        stresses = [0.0]*7
        stresses[0] = float(USUM1.Average.ToString()[:-5])
        stresses[1] = float(USUM2.Average.ToString()[:-5])
        stresses[2] = float(USUM3.Average.ToString()[:-5])
        stresses[3] = float(USUM4.Average.ToString()[:-5])
        stresses[4] = float(USUM5.Average.ToString()[:-5])
        stresses[5] = float(USUM6.Average.ToString()[:-5])
        stresses[6] = float(USUM7.Average.ToString()[:-5])
        string += str(i+1) + ',' + str(5*(f+1)) + ','
        for p in range(7):
            string += str(100000*stresses[p])[:10]+(p!=6)*',' + (p==6)*'\n'

        Load.Delete()
        USUM1.Delete()
        USUM2.Delete()
        USUM3.Delete()
        USUM4.Delete()
        USUM5.Delete()
        USUM6.Delete()
        USUM7.Delete()
print(string)