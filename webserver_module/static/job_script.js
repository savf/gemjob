var maxWidthMobile = 700;

$(document).ready(function() {
	adjustToSize();
	$( "#Datepicker" ).datepicker({ dateFormat: 'mm-dd-yy' });
	jobTypeSwitch()

	$(window).resize(function() {
		adjustToSize();
	});

});

function adjustToSize() {
    var isMobileSize = $(window).width() <= maxWidthMobile;

    if (isMobileSize){
        $(".Label").css({"max-width": "100%", "min-width": "100%"});
        $(".LabeledElement").css({"max-width": "100%", "min-width": "100%", "margin-left": "0%"});
    }
    else {
        $(".Label").css({"max-width": "25%", "min-width": "25%"});
        $(".LabeledElement").css({"max-width": "70%", "min-width": "70%", "margin-left": "5%"});
    }
}

function jobTypeSwitch() {
    if (document.getElementById("JobType").value == "hourly"){
        $(".IfHourly").show();
        $(".IfFixed").hide();
    }
    else{
        $(".IfHourly").hide();
        $(".IfFixed").show();
    }
}