package dao;

import java.util.ArrayList;

import dto.Product;

public class ProductRepository {

	private ArrayList<Product> listOfProducts = new ArrayList<Product>();

	public ProductRepository() {
		Product phone = new Product("1", "방울토마토 (샘플)");
		phone.setKitDate("2020년 5월 5일에 키트 설치됨");
		phone.setIllumination("70%");
		phone.setHumidity("50%");
		phone.setUnitsInStock(1500);
		phone.setCondition("정상");
		phone.setFilename("sample.jpg");

		listOfProducts.add(phone);
		}

	public ArrayList<Product> getAllProducts() {
		return listOfProducts;
	}	
	
	public Product getProductById(String productId) {
		Product productById = null;

		for (int i = 0; i < listOfProducts.size(); i++) {
			Product product = listOfProducts.get(i);
			if (product != null && product.getProductId() != null && product.getProductId().equals(productId)) {
				productById = product;
				break;
			}
		}
		return productById;
	}
}
