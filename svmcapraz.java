package crossValidation;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import java.util.Random;

public class svmcapraz {


	public static void main(String[] args) throws Exception {
		// loads data and set class index
		Instances data = DataSource.read("C:\\Users\\Erhan\\eclipse-workspace\\mermer_kalitelendirme\\src\\crossValidation\\mix4sHistLBP.arff");
	
		data.setClassIndex(data.numAttributes()-1);
	
		String crs="";
		for (int m = 0; m <12; m++) {
			SMO smo=new SMO();//NormalizedPolyKernel
			smo.setKernel(Kernel.forName("weka.classifiers.functions.supportVector.RBFKernel", null));//NormalizedPolyKernel-PolyKernel-Puk-RBFKernel
			String calibrasyon_ayar[]={"-N","1","-calibrator","weka.classifiers.functions.SMO"}; //kalibrasyon ve N=1 standardize
			smo.setOptions(calibrasyon_ayar);
			smo.setNumDecimalPlaces(2);
			int folds = 10;			
			smo.setC(0.25+0.25*m);
			Evaluation eval = new Evaluation(data);
			smo.buildClassifier(data);
			eval.crossValidateModel(smo, data, folds, new Random(1));

			System.out.println("--------------------------");			
			System.out.println(eval.toSummaryString());
			// output evaluation
			System.out.println();
			//System.out.println(eval.toMatrixString("=== Confusion matrix for fold " + (n+1) + "/" + folds + " ===\n"));							
			Double accuracy=eval.correct()/(eval.correct()+eval.incorrect());
			crs=crs+round((accuracy*100),2)+" ";			 
		}
		System.out.println("Capraz Dogrulama:"+crs);	 
		
	}
	public static double round(double value, int places) {
		if (places < 0) throw new IllegalArgumentException();

		long factor = (long) Math.pow(10, places);
		value = value * factor;
		long tmp = Math.round(value);
		return (double) tmp / factor;
	}
	public static double ortalamaHesapla(Double dizi[]){ 
		double toplam = 0; 
		for (int i = 0; i < dizi.length; i++){ 
			toplam = toplam + dizi[i]; 					//Formül ile ortalama deðerini bulma iþlemi 
		} 
		return (double) (toplam / dizi.length); 	//Ortalama  alýndý
	} 
	public static double standartSapmaHesapla(Double dizi[], double ort) { 
		double kareToplam = 0; 
		for (int i = 0; i < dizi.length; i++){ 
			kareToplam = kareToplam + dizi[i] * dizi[i]; 			//Formül ile standart sapma bulme iþlemi
		} 
		return (double) Math.sqrt(kareToplam / dizi.length - ort*ort);	//Sapma deðeri alýndý
	} 
}