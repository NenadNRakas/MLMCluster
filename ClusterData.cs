using Microsoft.ML.Data;

namespace MLMCluster
{
    //Source input data instance
    public class ClusterData
    {
        [LoadColumn(0)]
        public float SepalLength;
        [LoadColumn(1)]
        public float SepalWidth;
        [LoadColumn(2)]
        public float PetalLength;
        [LoadColumn(3)]
        public float PetalWidth;
    }
    // Resulting output binding
    public class ClusterPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId;
        [ColumnName("Score")]
        public float[]? Distances;
    }
}
