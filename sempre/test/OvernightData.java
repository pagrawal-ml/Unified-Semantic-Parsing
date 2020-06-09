package com.ibm.sempre.test;

public class OvernightData {

	public String utterance;
	public String original;
	public String targetFormula;
	
	public String getUtterance() {
		return utterance;
	}
	
	public void setUtterance(String utterance) {
		this.utterance = utterance;
	}
	public String getOriginal() {
		return original;
	}
	
	public void setOriginal(String original) {
		this.original = original;
	}
	
	public String getTargetFormula() {
		return targetFormula;
	}
	
	public void setTargetFormula(String targetFormula) {
		this.targetFormula = targetFormula;
	}
	
	@Override
	public String toString() {
		String data = "";
		data += "(example" + "\n";
		data += "  (utterance " + utterance + ")\n";
		data += "  (original " + original + ")\n";
		data += "  (targetFormula \n";
		data += "    " + targetFormula + "\n";
		data += "\n";
		data += "  )\n";
		data += ")\n";
		return data;
	}
}
