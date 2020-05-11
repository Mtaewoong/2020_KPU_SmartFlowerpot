package dto;

import java.io.Serializable;

public class Product implements Serializable {

	private static final long serialVersionUID = -4274700572038677000L;

	private String productId;	//primary key
	private String pname;		//화분 이름 (가칭)
	private String KitDate;		//키트 설치 날짜
	private String Humidity;	//습도
	private String Illumination;//조도
	private long unitsInStock;	//급수 시간 (
	private String condition;	//식물 상태 <건조, 정상, 포화>
	private String filename;	//사진

	public Product() {
		super();
	}

	public Product(String productId, String pname) {
		this.productId = productId;
		this.pname = pname;
	}

	public String getProductId() {
		return productId;
	}

	public String getPname() {
		return pname;
	}

	public void setPname(String pname) {
		this.pname = pname;
	}

	public void setProductId(String productId) {
		this.productId = productId;
	}

	public String getKitDate() {
		return KitDate;
	}

	public void setKitDate(String kitDate) {
		this.KitDate = kitDate;
	}

	public String getHumidity() {
		return Humidity;
	}

	public void setHumidity(String Humidity) {
		this.Humidity = Humidity;
	}

	public String getIllumination() {
		return Illumination;
	}

	public void setIllumination(String illumination) {
		this.Illumination = illumination;
	}

	public long getUnitsInStock() {
		return unitsInStock;
	}

	public void setUnitsInStock(long unitsInStock) {
		this.unitsInStock = unitsInStock;
	}

	public String getCondition() {
		return condition;
	}

	public void setCondition(String condition) {
		this.condition = condition;
	}
	
	public String getFilename() {
		return filename;
	}
	public void setFilename(String filename) {
		this.filename = filename;
	}
	
}
