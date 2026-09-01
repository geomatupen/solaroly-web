(function initializeGeoreferencedImageOverlay(){
  "use strict";

  if(!window.L) return;

  const GeoreferencedImageOverlay = window.L.Layer.extend({
    options: {
      opacity: 1,
      pane: "overlayPane",
      interactive: false,
      alt: "",
    },

    initialize(url, corners, options){
      window.L.setOptions(this, options);
      this._url = url;
      this._corners = corners.slice(0, 4).map(point => window.L.latLng(Number(point[1]), Number(point[0])));
    },

    onAdd(map){
      this._map = map;
      if(!this._image) this._initializeImage();
      this.getPane().appendChild(this._image);
      this._reset();
      return this;
    },

    onRemove(){
      this._image?.remove();
      this._map = null;
      return this;
    },

    getEvents(){
      return {
        zoom: this._reset,
        viewreset: this._reset,
        moveend: this._reset,
        zoomanim: this._animateZoom,
      };
    },

    getElement(){
      return this._image;
    },

    getBounds(){
      return window.L.latLngBounds(this._corners);
    },

    setOpacity(opacity){
      this.options.opacity = Number(opacity);
      if(this._image) window.L.DomUtil.setOpacity(this._image, this.options.opacity);
      return this;
    },

    _initializeImage(){
      const image = this._image = window.L.DomUtil.create("img", "leaflet-image-layer leaflet-zoom-animated");
      image.alt = this.options.alt || "";
      image.src = this._url;
      image.style.position = "absolute";
      image.style.maxWidth = "none";
      image.style.maxHeight = "none";
      image.style.transformOrigin = "0 0";
      image.style.pointerEvents = this.options.interactive ? "auto" : "none";
      window.L.DomUtil.setOpacity(image, this.options.opacity);
      image.addEventListener("load", () => {
        // Use the actual inference raster dimensions as the affine basis so
        // image pixels and inference-box pixels share the same coordinates.
        image.style.width = `${image.naturalWidth}px`;
        image.style.height = `${image.naturalHeight}px`;
        this._reset();
        this.fire("load");
      });
      image.addEventListener("error", event => this.fire("error", { error: event }));
    },

    _layerPoints(zoomEvent){
      if(zoomEvent && this._map?._latLngToNewLayerPoint){
        return this._corners.map(corner => (
          this._map._latLngToNewLayerPoint(corner, zoomEvent.zoom, zoomEvent.center)
        ));
      }
      return this._corners.map(corner => this._map.latLngToLayerPoint(corner));
    },

    _applyTransform(points){
      if(!this._image || points.length < 4) return;
      const topLeft = points[0];
      const topRight = points[1];
      const bottomLeft = points[3];
      const sourceWidth = Number(this._image.naturalWidth || this._image.width || 1);
      const sourceHeight = Number(this._image.naturalHeight || this._image.height || 1);
      const a = (topRight.x - topLeft.x) / sourceWidth;
      const b = (topRight.y - topLeft.y) / sourceWidth;
      const c = (bottomLeft.x - topLeft.x) / sourceHeight;
      const d = (bottomLeft.y - topLeft.y) / sourceHeight;
      this._image.style.transform = `matrix(${a},${b},${c},${d},${topLeft.x},${topLeft.y})`;
    },

    _reset(){
      if(this._map) this._applyTransform(this._layerPoints());
    },

    _animateZoom(event){
      if(this._map) this._applyTransform(this._layerPoints(event));
    },
  });

  window.GeoreferencedImageOverlay = GeoreferencedImageOverlay;
  window.createGeoreferencedImageOverlay = function(url, corners, options = {}){
    if(Array.isArray(corners) && corners.length >= 4){
      return new GeoreferencedImageOverlay(url, corners, options);
    }
    return window.L.imageOverlay(url, options.bounds, options);
  };
})();
