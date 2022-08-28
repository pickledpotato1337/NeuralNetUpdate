using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Globalization;

namespace NeuralNetUpdate
{

    public class Neuron
    {
        public float Bias { get; set; }
        public float SigmoidAdjustment { get; set; }

        public List<float> Weights { get; set; }
        public float Sigmoidize(float output)
        {
            
            double sig = 1 / (1 + Math.Exp((double)output));
            return (float)sig - SigmoidAdjustment;
        }
        public Neuron(float[] inputs)
        {
            SigmoidAdjustment = 0;
            Weights = new List<float>();
            for (int i = 0; i < inputs.Length - 1; i++)
            {

                Weights.Add(inputs[i]);

            }
            Bias = inputs[inputs.Length - 1];
        }

    }


    public class Set
    {
        public List<float> Values;
        public Set()
        {
            Values = new List<float>();
        }
    }

    public class NeuralLayer
    {

        public List<Neuron> neurons;
        public List<float> outputs;
        float[] Layer;
        public NeuralLayer entryPoint;
        public NeuralLayer outPoint;
        public NeuralLayer() {
            
            Layer = new float[1];
            neurons = new List<Neuron>();

        }
        public NeuralLayer(int size, List<Neuron> inputs)
        {
            neurons = inputs;
            Layer = new float[size];
        }
        public void SetEntry(NeuralLayer layer)
        {
            entryPoint = layer;
        }
        public void SetOut(NeuralLayer layer)
        {
            outPoint = layer;
        }
        public void SetLayerArray(float[] array)
        {
            Layer = array;
        }
        public void SetNeurons(List<Neuron> input) { neurons = input; }

        public List<float> PassThrough(List<float> entry)
        {
            List<float> res = new List<float>();

            for (int ct = 0; ct < neurons.Count; ct++)
            {
                float calc=0;
                for (int ctInner = 0; ctInner < entry.Count; ctInner++)
                {
                    
                    calc+= neurons[ct].Weights[ctInner] * entry[ctInner];
                }
                calc += neurons[ct].Bias;
                calc = neurons[ct].Sigmoidize(calc);
                res.Add(calc);
            }
            outputs = res;
            return res;

        }

        public List<Set> Train(List<Set> errorSets, float learningStep)
        {
            List<Set> result = new List<Set>();
            foreach (Set errorSet in errorSets)
            {
                
                List<float> errors = errorSet.Values;
                for (int j = 0; j < errors.Count; j++)
                {
                    float error = errors[j];
                    
                        
                            var OutputErrors = new List<float>();
                            error *= learningStep;
                            float sigAdj = error * outputs[j] * (1 - outputs[j]);
                            neurons[j].SigmoidAdjustment = sigAdj;
                            for (int k = 0; k < neurons[j].Weights.Count; k++)
                            {
                                if (k == neurons[j].Weights.Count - 1)
                                {
                                    error = sigAdj;
                                    
                                    neurons[j].Weights[k] += error;
                                }
                                else
                                {
                                    error = sigAdj * outputs[j];
                                    
                                    neurons[j].Weights[k] -= error;
                                    OutputErrors.Add(error);
                                }
                            }
                        var wrapper = new Set();
                        wrapper.Values = OutputErrors;
                        result.Add(wrapper);

                    

                    
                }
                
            }
            return result;
        }
    }

    public class NetworkCreator
    {
        public List<NeuralLayer> inputs;
        public List<string> paths;
        public NetworkCreator(List<int> layers, List<Set> inputData)
        {
            string pathBase = "layer";
            string pathFormat = ".txt";
            inputs = new List<NeuralLayer>();
            paths = new List<string>();
            for (int i = 0; i < layers.Count; i++)
            {
                string layerPath = pathBase + i.ToString() + pathFormat;
                paths.Add(layerPath);
                if (i == 0)
                {
                    ReadAndWrite.CreateNeurons(layerPath, inputData[0].Values.Count, layers[i]);
                }
                if (i > 0)
                {
                    ReadAndWrite.CreateNeurons(layerPath, layers[i - 1], layers[i]);
                }
                var aux = ReadAndWrite.LoadNeurons(layerPath);
                var input = new NeuralLayer(layers[i], aux);
                inputs.Add(input);
                if (i > 0)
                {
                    inputs[i].SetEntry(inputs[i - 1]);
                    if (i < layers.Count - 1) {
                        inputs[i - 1].SetOut(inputs[i]); }
                }

            }


        }
        public List<float> GetResult(Set set)
        {

            List<float> result = new List<float>();
            
                var entry = set.Values;
                foreach (NeuralLayer layer in inputs)
                {

                    var dispatch = layer.PassThrough(entry);
                    entry = dispatch;
                    result = dispatch;
                }
            
            return result ;
        }
        public void Train(List<Set> inputData, float learningStep, int LearningLoops, int learningPacket, List<Set> demandedResult)
        {
            
            for (int ctr = 0; ctr < LearningLoops; ctr++)
            {   
                for(int setCtr=0; setCtr< demandedResult.Count; setCtr++) { 

                     var result = GetResult(inputData[setCtr]);
                    var errorEntry = new List<Set>();
                    var wrapper = new Set();

                    inputs.Reverse();
                    for (int i = 0; i < demandedResult[setCtr].Values.Count; i++)
                    {
                        result[i] -= demandedResult[setCtr].Values[i];

                    }
                    wrapper.Values = result;
                    errorEntry.Add(wrapper);
                    int inner = 0;
                    foreach (NeuralLayer layer in inputs)
                    {
                        
                        var substitute=layer.Train(errorEntry, learningStep);
                        errorEntry = substitute;
                        //needs fix, error for last layer in entire network
                        if (ctr % learningPacket == 0)
                        {
                            ReadAndWrite.SaveNeurons(layer.neurons, paths[inner]);
                        }
                        inner++;
                    }
                    inputs.Reverse();
                }
               
            }
        }
    }
    public class ReadAndWrite
    {
        public static List<Neuron> LoadNeurons(string path)
        {
            List<Neuron> result = new List<Neuron>();
            string[] readText = File.ReadAllLines(path);
            foreach (string line in readText)
            {
                string separator = "\t";
                string[] variables = line.Split(separator.ToCharArray());
                float[] parameters = new float[variables.Length];
                for (int i = 0; i < variables.Length; i++)
                {
                    parameters[i] = float.Parse(variables[0], CultureInfo.InvariantCulture);

                }
                var NetElement = new Neuron(parameters);
                result.Add(NetElement);
            }

            return result;
        }
        public static void SaveNeurons(List<Neuron> neurons, string path)
        {
            using (StreamWriter outfile = new StreamWriter(path))
            {
                foreach (Neuron elem in neurons)
                {
                    string content = "";
                    foreach (float value in elem.Weights)
                    {
                        content += value.ToString() + "\t";
                    }
                    outfile.WriteLine(content);
                }
            }
        }


        public static void CreateNeurons(string path1, int inputCount, int layerCount)
        {
            float[,] firstLayer = new float[layerCount, inputCount + 1];

            Random rnd = new Random();
            for (int i = 0; i < layerCount; i++)
            {
                for (int j = 0; j <= inputCount; j++)
                {
                    firstLayer[i, j] = (float)rnd.Next(1, 999);
                    firstLayer[i, j] /= 1000;
                }
            }


            using (StreamWriter outfile1 = new StreamWriter(path1))
            {
                for (int i = 0; i < layerCount; i++)
                {

                    string content = "";
                    for (int j = 0; j <= inputCount; j++)
                    {
                        content += firstLayer[i, j].ToString() + "\t";
                    }
                    outfile1.WriteLine(content);
                }
            }



        }

    }

    class Program
    {
        static void Main(string[] args)
        {
            int state = 1;
            Console.WriteLine("podaj rozmiary warstw");
            string line = Console.ReadLine();
            string separator = " ";
            string[] variables = line.Split(separator.ToCharArray());
            List<int> layerSizes = new List<int>();
            for (int i = 0; i < variables.Length; i++)
            {
                var count = Int32.Parse(variables[i]);
                layerSizes.Add(count);
            }

            Console.WriteLine("podaj ilość zestawów");
            string sets = Console.ReadLine();
            int setsCount = Int32.Parse(sets);
            List<Set> inputs = new List<Set>();
            for (int j = 0; j < setsCount; j++)
            {
                var set = new List<float>();
                Console.WriteLine("podaj wejscia");
                var setstring = Console.ReadLine();
                string[] variables2 = setstring.Split(separator.ToCharArray());
                for (int i = 0; i < variables2.Length; i++)
                {
                    var count = float.Parse(variables2[i]);
                    set.Add(count);
                    var wrapper = new Set();
                    wrapper.Values = set;
                    inputs.Add(wrapper);

                }
            }
            var creator = new NetworkCreator(layerSizes, inputs);

            while (state > 0)
            {
                Console.WriteLine("1-trenuj, 2-uzyskaj wynik, 0-wyjdz");
                string stateStr = Console.ReadLine();
                state = Int32.Parse(stateStr);

                if (state == 1)
                {
                    List<Set> outputs = new List<Set>();
                    for (int j = 0; j < setsCount; j++)
                    {
                        var set = new List<float>();
                        Console.WriteLine("podaj wyjscia oczekiwane");
                        var setstring = Console.ReadLine();
                        string[] variables2 = setstring.Split(separator.ToCharArray());
                        for (int i = 0; i < variables2.Length; i++)
                        {
                            var count = float.Parse(variables2[i]);
                            set.Add(count);
                            var wrapper = new Set();
                            wrapper.Values = set;
                            outputs.Add(wrapper);

                        }
                    }
                    Console.WriteLine("podaj ilosć petli nauczania");
                    var loopString= Console.ReadLine();
                    int loops = Int32.Parse(loopString);

                    Console.WriteLine("podaj rozmiar pakietu");
                    var packetString= Console.ReadLine();
                    int packet = Int32.Parse(packetString);

                    Console.WriteLine("podaj rozmiar kroku nauczania");
                    var stepString= Console.ReadLine();
                    float step = float.Parse(stepString, CultureInfo.InvariantCulture);


                    creator.Train(inputs, step, loops, packet, outputs);
                }
                if (state == 2)
                {
                    List<float> pass = creator.GetResult(inputs[0]);
                    foreach (float value in pass)
                    {
                        Console.WriteLine(value);
                    }
                }

                Console.ReadKey();

            }
        }
    }
}
