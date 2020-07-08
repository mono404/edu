using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace edu.LightGamEx
{
    class LightGbmEx
    {
        public LightGbmEx(string pathname /*= "creditcard.csv"*/, string modelname/* = "model.zip"*/)
        {
            MLContext mlContext = new MLContext();

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: pathname,
                                            hasHeader: true,
                                            separatorChar: ',',
                                            allowQuoting: true,
                                            allowSparse: false);

            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", new[] { "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount" });

            // Choosing algorithm
            var trainer = mlContext.BinaryClassification.Trainers.LightGbm(labelColumnName: "Class", featureColumnName: "Features");
            // Appending algorithm to pipeline
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            ITransformer model = trainingPipeline.Fit(trainingDataView);
            mlContext.Model.Save(model, trainingDataView.Schema, modelname);

            var crossValidationResults = mlContext.BinaryClassification.CrossValidateNonCalibrated(trainingDataView, trainingPipeline, numberOfFolds: 5, labelColumnName: "Class");

            Console.WriteLine(crossValidationResults);

            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

            ModelInput sampleData = new ModelInput()
            {
                Time = 0,
                V1 = -2.076174782f,
                V2 = 2.142237995f,
                V3 = -2.522703577f,
                V4 = -1.888063034f,
                V5 = 1.98278475f,
                V6 = 3.732949553f,
                V7 = -1.217430393f,
                V8 = -0.536644267f,
                V9 = 0.272867038f,
                V10 = 0.300342205f,
                V11 = -0.451655998f,
                V12 = 0.566367644f,
                V13 = -0.317804444f,
                V14 = 0.855741736f,
                V15 = -0.041046986f,
                V16 = 0.046620056f,
                V17 = 0.01782216f,
                V18 = -0.772915626f,
                V19 = -0.354162802f,
                V20 = -0.308523004f,
                V21 = 2.016666112f,
                V22 = -1.588268798f,
                V23 = 0.588482263f,
                V24 = 0.632443919f,
                V25 = -0.201063916f,
                V26 = 0.199251167f,
                V27 = 0.43865731f,
                V28 = 0.172923188f,
                Amount = 8.95f,
                Class = false
            };

            ModelOutput predictionResult = predEngine.Predict(sampleData);

            Console.WriteLine($"Actual value: {sampleData.Class} | Predicted value: {predictionResult.Prediction}");
        }

        public class ModelInput
        {
            [ColumnName("Time"), LoadColumn(0)]
            public float Time { get; set; }

            [ColumnName("V1"), LoadColumn(1)]
            public float V1 { get; set; }

            [ColumnName("V2"), LoadColumn(2)]
            public float V2 { get; set; }

            [ColumnName("V3"), LoadColumn(3)]
            public float V3 { get; set; }

            [ColumnName("V4"), LoadColumn(4)]
            public float V4 { get; set; }

            [ColumnName("V5"), LoadColumn(5)]
            public float V5 { get; set; }

            [ColumnName("V6"), LoadColumn(6)]
            public float V6 { get; set; }

            [ColumnName("V7"), LoadColumn(7)]
            public float V7 { get; set; }

            [ColumnName("V8"), LoadColumn(8)]
            public float V8 { get; set; }

            [ColumnName("V9"), LoadColumn(9)]
            public float V9 { get; set; }

            [ColumnName("V10"), LoadColumn(10)]
            public float V10 { get; set; }

            [ColumnName("V11"), LoadColumn(11)]
            public float V11 { get; set; }

            [ColumnName("V12"), LoadColumn(12)]
            public float V12 { get; set; }

            [ColumnName("V13"), LoadColumn(13)]
            public float V13 { get; set; }

            [ColumnName("V14"), LoadColumn(14)]
            public float V14 { get; set; }

            [ColumnName("V15"), LoadColumn(15)]
            public float V15 { get; set; }

            [ColumnName("V16"), LoadColumn(16)]
            public float V16 { get; set; }

            [ColumnName("V17"), LoadColumn(17)]
            public float V17 { get; set; }

            [ColumnName("V18"), LoadColumn(18)]
            public float V18 { get; set; }

            [ColumnName("V19"), LoadColumn(19)]
            public float V19 { get; set; }

            [ColumnName("V20"), LoadColumn(20)]
            public float V20 { get; set; }

            [ColumnName("V21"), LoadColumn(21)]
            public float V21 { get; set; }

            [ColumnName("V22"), LoadColumn(22)]
            public float V22 { get; set; }

            [ColumnName("V23"), LoadColumn(23)]
            public float V23 { get; set; }

            [ColumnName("V24"), LoadColumn(24)]
            public float V24 { get; set; }

            [ColumnName("V25"), LoadColumn(25)]
            public float V25 { get; set; }

            [ColumnName("V26"), LoadColumn(26)]
            public float V26 { get; set; }

            [ColumnName("V27"), LoadColumn(27)]
            public float V27 { get; set; }

            [ColumnName("V28"), LoadColumn(28)]
            public float V28 { get; set; }

            [ColumnName("Amount"), LoadColumn(29)]
            public float Amount { get; set; }

            [ColumnName("Class"), LoadColumn(30)]
            public bool Class { get; set; }
        }

        class ModelOutput
        {
            [ColumnName("PredictedLabel")]
            public bool Prediction { get; set; }

            public float Score { get; set; }
        }
    }
}
