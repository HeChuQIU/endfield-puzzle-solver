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
    }
};
