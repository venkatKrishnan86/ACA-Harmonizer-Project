const {defineConfig} = require('@vue/cli-service')
module.exports = defineConfig({
    devServer: {
        proxy: 'http://127.0.0.1:5000'
    },
    transpileDependencies: true
})
