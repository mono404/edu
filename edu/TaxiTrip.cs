using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;

namespace edu
{
    
    class TaxiFarePrediction
    {
        // <Snippet2>
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "train.csv");
        static readonly string _valDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "test.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        // </Snippet2>

        public TaxiFarePrediction()
        {
            Console.WriteLine(Environment.CurrentDirectory);

            MLContext mlContext = new MLContext(seed: 0);

            var model = Train(mlContext, _trainDataPath, _valDataPath);

            Evaluate(mlContext, model);

            TestSinglePrediction(mlContext, model);
        }

        public static ITransformer Train(MLContext mlContext, string trainDataPath, string valDataPath)
        {
            IDataView trainDataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(trainDataPath, hasHeader: true, separatorChar: ',');
            IDataView valDataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(valDataPath, hasHeader: true, separatorChar: ',');

            // regression 사용
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                    .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
                    // labelColumnName, featureColumnName, exampleWeightCloynmName, numberOfLeaves, numberOfTrees, minimumExampleCountPerLeaf, learningRate
                    .Append(mlContext.Regression.Trainers.FastTree("Label", "Features", null, 50, 200, 30, 0.2)
                    );


            Console.WriteLine("=============== Create and Train the Model ===============");
            
            /*   regression 사용   */
            var model = pipeline.Fit(trainDataView);

            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            // 모델 저장
            mlContext.Model.Save(model, trainDataView.Schema, "TaxiFarePridictionModel.zip");

            return model;
        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView traindataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: ',');
            IDataView valdataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_valDataPath, hasHeader: true, separatorChar: ',');

            var predictions = model.Transform(traindataView);
            var regMetrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       RSquared Score:      {regMetrics.RSquared:0.##}");
            Console.WriteLine($"*       Root Mean Squared Error:      {regMetrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"*************************************************");
            Console.WriteLine();

        }

        private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            //Prediction test
            // Create prediction function and make prediction.
            var predictionFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);
            //Sample:
            //vendor_id,rate_code,passenger_count,trip_time_in_secs,trip_distance,payment_type,fare_amount
            //VTS,1,1,1140,3.75,CRD,15.5
            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            };

            var prediction = predictionFunction.Predict(taxiTripSample);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");
        }

        private static void PrintMetrics(BinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"Accuracy: {metrics.Accuracy:F2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:F2}");
            Console.WriteLine($"Negative Precision: " +
                $"{metrics.NegativePrecision:F2}");

            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall:F2}");
            Console.WriteLine($"Positive Precision: " +
                $"{metrics.PositivePrecision:F2}");

            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:F2}\n");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
        }

        public class TaxiTrip
        {
            [LoadColumn(0)]
            public string VendorId;

            [LoadColumn(1)]
            public string RateCode;

            [LoadColumn(2)]
            public float PassengerCount;

            [LoadColumn(3)]
            public float TripTime;

            [LoadColumn(4)]
            public float TripDistance;

            [LoadColumn(5)]
            public string PaymentType;

            [LoadColumn(6)]
            public float FareAmount;
        }

        public class TaxiTripFarePrediction
        {
            [ColumnName("Score")]
            public float FareAmount;
        }


    }

    
}
