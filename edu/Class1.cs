using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML;

namespace edu
{
    
    class Class1
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        public Class1()
        {
            MLContext mlContext = new MLContext(seed: 0);

            var model = Train(mlContext, _trainDataPath);
        }

        

        
    }

    public static ITransformer Train(MLContext mlContext, string dataPath)
    {
        IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');

        var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")

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
