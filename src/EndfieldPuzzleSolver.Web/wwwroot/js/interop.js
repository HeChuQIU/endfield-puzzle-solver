window.blazorInterop = {
    clickElement: function (elementId) {
        var el = document.getElementById(elementId);
        if (el) el.click();
    },
    getClipboardImage: async function () {
        try {
            var items = await navigator.clipboard.read();
            for (var i = 0; i < items.length; i++) {
                var item = items[i];
                for (var j = 0; j < item.types.length; j++) {
                    var type = item.types[j];
                    if (type.startsWith('image/')) {
                        var blob = await item.getType(type);
                        var buffer = await blob.arrayBuffer();
                        return new Uint8Array(buffer);
                    }
                }
            }
        } catch (e) {
            console.warn('Clipboard access failed:', e);
        }
        return null;
    },
    setupDropZone: function (element, dotNetHelper) {
        if (!element) return;
        
        element.addEventListener('drop', async function (e) {
            e.preventDefault();
            e.stopPropagation();
            
            if (!e.dataTransfer || !e.dataTransfer.files || e.dataTransfer.files.length === 0) {
                return;
            }
            
            var file = e.dataTransfer.files[0];
            if (!file.type.startsWith('image/')) {
                await dotNetHelper.invokeMethodAsync('OnDropError', '请拖入图片文件（PNG/JPG）');
                return;
            }
            
            try {
                var buffer = await file.arrayBuffer();
                var bytes = new Uint8Array(buffer);
                await dotNetHelper.invokeMethodAsync('OnDropFile', bytes, file.name);
            } catch (error) {
                console.error('Drop file error:', error);
                await dotNetHelper.invokeMethodAsync('OnDropError', '读取文件失败');
            }
        });
        
        element.addEventListener('dragover', function (e) {
            e.preventDefault();
            e.stopPropagation();
        });
    }
};
