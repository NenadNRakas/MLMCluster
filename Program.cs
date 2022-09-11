using Microsoft.ML;
using MLMCluster;

namespace MLMCluster
{
    internal class Program
    {
        static void Main(string[] args)
        {
            string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data");
            string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");
            //Console.WriteLine("an [@.i.]™ World!");
            // Create a context
            var mlContext = new MLContext(seed: 0);
            //mlContext.Run(args);
            // Data loading setup
            IDataView dataView = mlContext.Data.LoadFromTextFile<ClusterData>(_dataPath, hasHeader: false, separatorChar: ',');
            // Create a pipeline
            string featuresColumnName = "Features";
            var pipeline = mlContext.Transforms
                .Concatenate(featuresColumnName, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3));
            //var model = mlContext.Transforms;
            //model.Transforms.Add();
            // Train the model
            var model = pipeline.Fit(dataView);
            // Save the model
            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, dataView.Schema, fileStream);
            }
            // Run a model based prediction            
            var predictor = mlContext.Model.CreatePredictionEngine<ClusterData, ClusterPrediction>(model);
            var prediction = predictor.Predict(TestData.Setosa);
            Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
            if (prediction.Distances != null)
            {
                Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");
            }
        }
    }
}
