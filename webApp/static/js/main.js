(function () {
    "use strict";

    var page = {

        ready: function () {
            var formdata = {};
            $("#processresultdiv").hide();

            // base upload function
            $('#uploadImage').fileinput({
                uploadUrl: '/uploadImage',
                theme : 'explorer-fas',
                uploadAsync: false,
                showUpload: false,
                showRemove :true,
                showPreview: true,
                showCancel:true,
                showCaption: true,
                maxFileCount: 1,
                allowedFileExtensions: ['jpg', 'png'],
                uploadExtraData: function(previewId, index) {
                    return formdata
                },
                browseClass: "btn btn-primary ",
                dropZoneEnabled: true,
                dropZoneTitle: 'Drag file here！',
            });

            // Face Encoding upload button
            $(".btn-upload-3").on("click", function() {
                $("#processresultdiv").hide();
                var username = $("#username").val();
                if(!username) {
                    alert('Please Enter Name.');
                    return;
                }

                formdata = {
                  "username": $("#username").val()
                }

                $("#uploadImage").fileinput('upload');
            });

            // Face Encoding clear button
            $(".btn-reset-3").on("click", function() {
                $("#username").val('');
                $("#uploadImage").fileinput('clear');
            });

            // call back function for upload Image file
            $('#uploadImage').on('fileuploaded', function(event, data, previewId, index) {
                $("#username").val('');
            });

            $('#uploadFile').fileinput({
                uploadUrl: '/uploadfile',
                theme : 'explorer-fas',
                uploadAsync: false,
                showUpload: false,
                showRemove :true,
                showPreview: true,
                showCancel:true,
                showCaption: true,
                allowedFileExtensions: ['jpg', 'png', 'mp4', 'avi', 'dat', '3gp', 'mov', 'rmvb'],
                maxFileSize : 153600,
                maxFileCount : 1,
                browseClass: "btn btn-primary ",
                dropZoneEnabled: true,
                dropZoneTitle: 'Drag file here！'
            });

            // image process upload button
            $(".btn-uploadfile-3").on("click", function() {
                $("#processresultdiv").hide();
                $("#uploadFile").fileinput('upload');
            });

            // image process clear button
            $(".btn-resetfile-3").on("click", function() {
                $("#uploadFile").fileinput('clear');
            });

            // call back function for upload file
            $('#uploadFile').on('filebatchuploadsuccess', function(event, data, previewId, index) {
                var url = "/download/"+data.response.filename;
                $("#downloadbtn").attr("href", url);
                $("#processresultdiv").show();
            });

            //show detected face
            $('body').on('DOMSubtreeModified', '.progress-bar-success', function(data){
                if ($('.progress-bar-success').length>0 && $('.progress-bar-success')[0].outerText =="Done") {
                    var uploadfiles = $('.file-footer-caption');
                    var fileName = $('.file-footer-caption')[uploadfiles.length - 1].title

                    var arr = fileName.split('.')
                    arr[0]=arr[0]+"_face"
                    var processedFileName = arr.join('.')
                    $("#uploadFile_processed").attr("src","/static/processed/"+processedFileName)
                    $("#processresultdiv").show();
                }
            });

            //show proceesed image
          $('body').on('DOMSubtreeModified', '.progress-bar-success', function(data){
              debugger
            if ($('.progress-bar-success').length>0 && $('.progress-bar-success')[0].outerText =="Done") {
                var uploadfiles = $('.file-footer-caption');
                var fileName = $('.file-footer-caption')[uploadfiles.length - 1].title

                var arr = fileName.split('.')
                arr[0]=arr[0]+"_processed"
                var processedFileName = arr.join('.')
                if (['JPG', 'PNG'].indexOf(arr[1].toUpperCase())>=0){
                    $("#uploadImg_processed").attr("src","/static/processed/"+processedFileName)
                    $("#processed_img_wrapper").attr("style","")
                }
                else {
                    $("#processed_img_wrapper").attr("style","display:none;")
                }

                $("#processresultdiv").show();
            }
          });

        }
    }

    $(document).ready(page.ready);

})();
