namespace Frontend.Generated
{
    public partial class KahflaneApiClient
    {
        static void UpdateJsonSerializerSettings(System.Text.Json.JsonSerializerOptions settings)
        {
            settings.PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase;
            settings.DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull;
        }
    }
}
