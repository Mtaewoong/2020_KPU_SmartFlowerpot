<%@ page contentType="text/html; charset=utf-8"%>
<%@ page import="dto.Product"%>
<jsp:useBean id="productDAO" class="dao.ProductRepository" scope="session" />
<html>
<head>
<link rel="stylesheet" href="./css/bootstrap.min.css" />
<title>상품 상세 정보</title>
</head>
<body>
	<%@ include file="menu.jsp"%>
	<div class="jumbotron">
		<div class="container">
			<h1 class="display-2">화분 정보</h1>
		</div>
	</div>
	<%
		String id = request.getParameter("id");
		Product product = productDAO.getProductById(id);
	%>
	<div class="container">
		<div class="row">
			<div class="col-md-5">
				<img src="./image/<%=product.getFilename()%>"style="width: 100%">
			</div>
			<div class="col-md-7">
				<h3><%=product.getPname()%></h3>
				<p><%=product.getKitDate()%>
				<p><b>상태: </b><span class="badge badge-danger"> <%=product.getCondition()%></span>
				<p><b>습도</b> : <%=product.getHumidity()%>
				<p><b>조도</b> : <%=product.getIllumination()%>
				<p><b>마지막 급수 시각</b> : <%=product.getUnitsInStock()%>초 전
				<p><b>다음 급수 예정 시각</b> : <%=product.getUnitsInStock() + 654%>초 후
			</div>
		</div>
		<hr>
	</div>
	<jsp:include page="footer.jsp" />
</body>
</html>
