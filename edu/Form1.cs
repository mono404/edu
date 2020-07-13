using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace edu
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();

            //TaxiFarePrediction taxiFarePrediction = new TaxiFarePrediction();
            //ResNetv2 resNetv2 = new ResNetv2();
            //Onnx.Onnx onnx = new Onnx.Onnx();
            LogisticRegression.SentimentAnalysis sentimentAnalysis = new LogisticRegression.SentimentAnalysis();
        }
    }
}
