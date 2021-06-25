var chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD"),
    chirps2 = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY"),
    Riferimento = ee.Geometry.Polygon([[[59.32339589402671, 46.96184431991006],
                                        [59.32339589402671, 44.052415375085985],
                                        [63.87173573777671, 44.052415375085985],
                                        [63.87173573777671, 46.96184431991006]]], null, false),
    table = ee.FeatureCollection("users/lorenzinspace_astropi/data_clean"),
    table2 = ee.FeatureCollection("users/lorenzinspace_astropi/datiFoto");

// Scatola nera che restituisce la pioggia
var pioggia = function(date, region) {
    var stats = chirps2.filter(ee.Filter.date(date, date.advance(1, 'day')))
      .reduce(ee.Reducer.sum()).reduceRegion({
    reducer: ee.Reducer.mean(), //media nell'immagine che è già la somma
    geometry: region,
    scale: 5000, //scala nominale in metri
    });
    return stats.get('precipitation_sum');
};
  
// Crea una lista alcui indice i corrisponde una lista che contiene lng e lat della foto i
var listacentri = [];
for (var i = 1000; i < 1340; i++){
    var image = table2.filter(ee.Filter.equals('Photo_id', i)).first();
    listacentri[i-1000] = [image.get('Longitude'), image.get('Latitude')];
}
  
// Trova le regioni corrispondenti alle foto sulla mappa
var yoff = 1.4547144724120358;
var xoff = 2.274169921875;
var get_region = function(center){
    // Questi tipi di variabile sono immondi
    return ee.Geometry.Rectangle([ee.Number(center[0]).subtract(xoff), 
                                  ee.Number(center[1]).subtract(yoff), 
                                  ee.Number(center[0]).add(xoff), 
                                  ee.Number(center[1]).add(yoff)]);
};
  
// Crea una lista dei valori in mm di pioggia corrispondenti alle regioni
var listapioggia = [];
for (var i=0; i < listacentri.length; i++) {
    listapioggia[i] = ee.Feature(null, {'mm_di_pioggia': pioggia(ee.Date.fromYMD(2021, 4, 21), get_region(listacentri[i]))});
    //listapioggia[i] = ee.Feature(null, {'mm_di_pioggia': pioggia(ee.Date.fromYMD(2019, 4, 15), get_region(listacentri[i+66]))});
}
//print(ee.FeatureCollection(listapioggia));
  
// Esporta la collezione
Export.table.toDrive({
    collection: ee.FeatureCollection(listapioggia),
    folder: 'test_fra',
    fileNamePrefix: 'pioggia_test02',
    fileFormat: 'CSV'});