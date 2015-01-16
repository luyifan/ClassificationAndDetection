$(document).ready(function($){

	$('#origin_image').qtip({
		content: {
			text: $('#origin_image').next('div')  
		},
	    	position: {
			adjust: { screen: true },	
			corner: {
				target:  'topRight',	
				tooltip: 'bottomLeft'
			}
		}
	})
	$("#prediction").mouseover(function(){

		console.log($(this).attr('value'))
	})
	
})


    


