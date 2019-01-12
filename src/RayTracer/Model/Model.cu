#include "Model.cuh"
#define UP_VEC make_float3(0.0f, getRadius(), 0.0f)

CUDA_FUNC float mix_produt(float3 a, float3 b, float3 c)
{
    return dot(cross(a, b), c);
}

CUDA_FUNC Triangle::Triangle(float3 a, float3 b, float3 c, float3 norma, float3 normb, float3 normc)
{
    pos[0] = a;
    pos[1] = b;
    pos[2] = c;
    normal[0] = norma;
    normal[1] = normb;
    normal[2] = normc;
}
CUDA_FUNC Triangle::Triangle(float2 t[3],float3 p[3], float3 norm[3]) {
	vText[0] = t[0];
	vText[1] = t[1];
	vText[2] = t[2];
	pos[0] = p[0];
	pos[1] = p[1];
	pos[2] = p[2];
	normal[0] = norm[0];
	normal[1] = norm[1];
	normal[2] = norm[2];
}
CUDA_FUNC Triangle::Triangle(float3 p[3], float3 norm[3]) {
    pos[0] = p[0];
    pos[1] = p[1];
    pos[2] = p[2];
	normal[0] = norm[0];
    normal[1] = norm[1];
    normal[2] = norm[2];
}
CUDA_FUNC Triangle::Triangle(const float3 p[3], const float3 norm[3]) {
    pos[0] = p[0];
    pos[1] = p[1];
    pos[2] = p[2];
    normal[0] = norm[0];
    normal[1] = norm[1];
    normal[2] = norm[2];
}
CUDA_FUNC Triangle::Triangle(float3 a, float3 b, float3 c, float3 norm[3]) {
    pos[0] = a;
    pos[1] = b;
    pos[2] = c;
    normal[0] = norm[0];
    normal[1] = norm[1];
    normal[2] = norm[2];
}


CUDA_FUNC float3 Triangle::interpolatePosition(float3 sample) const
{
    return transformation(sample.x * pos[0] + sample.y * pos[1] + sample.z * pos[2]);
}


CUDA_FUNC  bool  Triangle::hit(Ray r, IntersectRecord &colideRec) {

    //colideRec.t = -1.0f;
	float3 ta=transformation(pos[0]), tb=transformation(pos[1]), tc= transformation(pos[2]);
	

	float t;
    float3 norma = cross(ta - tc, tb - tc);
    float dot_normal_dir = dot(norma, r.getDir());
    if (fabs(dot_normal_dir) < FLOAT_EPISLON)
        return false;

    t = (-dot(r.getOrigin(), norma) + dot(ta, norma))/ dot_normal_dir;

    float3 pos = r.getPos(t);

    float S = area();
    float s1 = length(cross(pos - ta, pos -tb));
    float s2 = length(cross(pos - tc, pos - ta));
    float s3 = length(cross(pos - tc, pos - tb));
	
	float2 pvText;

    if (fabs(s1 + s2 + s3 - S) > 0.001f)
        return false;

    float m1 = s3 / S, m2 = s2 / S, m3 = 1.0f - m1 - m2;
	pvText = vText[0] * m1 + vText[1] * m2 + vText[2] * m3;
	//Todo: load this to colideRec
    if (t > FLOAT_EPISLON && t < colideRec.t)
    {
        colideRec.material = my_material;
        colideRec.material_type = material_type;
        colideRec.t = t;
        colideRec.normal = normalize(m1 * normal[0] + m2 * normal[1] + m3 * normal[2]);
        colideRec.pos = r.getPos(t);
		colideRec.isLight = false;
        colideRec.tangent =  normalize( cross(colideRec.normal, make_float3(0.3, 0.4, -0.5)));
		return true;
    }

    return false;
}

__host__  bool Mesh::readFile(char * path) {
	ifstream file(path);

	vector<float3> vVertex;
	vector<float2> vText;
	vector<float3> vNorm;
	vector<vector<int3>> vFace;
	if (!file) {
		return false;
	}
	string line;
	while (getline(file, line)) {
		if (line.substr(0, 2) == "vt") {
			istringstream s(line.substr(2));
			float2 v;
			s >> v.x; s >> v.y;

			v.y = -v.y;
			vText.push_back(v);
		}
		else if (line.substr(0, 2) == "vn") {
			istringstream s(line.substr(2));
			float3 v;
			s >> v.x; s >> v.y; s >> v.z;

			vNorm.push_back(v);
		}
		else if (line.substr(0, 1) == "v") {
			istringstream s(line.substr(1));
			float3 v;
			s >> v.x; s >> v.y; s >> v.z;

			vVertex.push_back(v);
		}
		else if (line.substr(0, 1) == "f") {
			vector <int3> face;

			istringstream vtns(line.substr(1));
			string vtn;
			while (vtns >> vtn) {
				int3 vertex;
				replace(vtn.begin(), vtn.end(), '/', ' ');
				istringstream ivtn(vtn);
				if (vtn.find("  ") != string::npos) {
					ivtn >> vertex.x >> vertex.y;

					vertex.x--;
					vertex.y--;
					vertex.z = 0xfff;
				}
				else {
					ivtn >> vertex.x
						>> vertex.y
						>> vertex.z;


					vertex.x--;
					vertex.y--;
					vertex.z--;
				}
				face.push_back(vertex);
			}
			vFace.push_back(face);

		}
		else if (line[0] == '#') {
		}
		else {

		}
	}
	if (vFace.empty())
		return false;
	//vector<Triangle> tempMesh;

	if (vText.size() != 0) {

		cudaMalloc((void**)& meshTable, sizeof(Triangle)*(vFace.size() + 1));

		Triangle *temp = (Triangle*)malloc(sizeof(Triangle)*(vFace.size() + 1));
		for (int f = 0; f < vFace.size(); f++) {
			int n = vFace[f].size();

			float3 V[3], N[3];
			float2 T[3];
			for (int v = 0; v < n; v++) {
				int it = vFace[f][v].z;
				if (vText.size() > 0) {
					T[v].x = vText[it].x;
					T[v].y = vText[it].y;
				}
				//	glTexCoord2f(vText[it].x, vText[it].y);

				int in = vFace[f][v].y;
				if (vNorm.size() > 0) {
					V[v].x = vNorm[in].x;
					V[v].y = vNorm[in].y;
					V[v].z = vNorm[in].z;
				}
				int iv = vFace[f][v].x;
				N[v].x = vVertex[iv].x;
				N[v].y = vVertex[iv].y;
				N[v].z = vVertex[iv].z;
				//	glVertex3f(vVertex[iv].x, vVertex[iv].y, vVertex[iv].z);
			}

			Triangle t(T,V,N);

			t.setUpTransformation(transformation);
			temp[f] = t;
			//	glEnd();
		}
		cudaMemcpy(meshTable, temp, sizeof(Triangle)*(vFace.size() + 1), cudaMemcpyHostToDevice);

	}

	else {

		cudaMalloc((void**)& meshTable, sizeof(Triangle)*(vFace.size() + 1));
		Triangle *temp = (Triangle*)malloc(sizeof(Triangle)*(vFace.size() + 1));
		for (int f = 0; f < vFace.size(); f++) {
			int n = vFace[f].size();
			float3 V[3], N[3];
			for (int v = 0; v < n; v++) {
				int in = (vFace[f])[v].y;
				int iv = (vFace[f])[v].x;
				if (vNorm.size() > 0) {
					V[v].x = vNorm[in].x;
					V[v].y = vNorm[in].y;
					V[v].z = vNorm[in].z;
				}
				N[v].x = vVertex[iv].x;
				N[v].y = vVertex[iv].y;
				N[v].z = vVertex[iv].z;
			}
			Triangle t(N,V);
			t.setUpTransformation(transformation);
			temp[f] = t;
		}
		cudaMemcpy(meshTable, temp, sizeof(Triangle)*(vFace.size() + 1), cudaMemcpyHostToDevice);
		
	}
	number = vFace.size();
	return true;
}


CUDA_FUNC  bool  Mesh::hit(Ray r, IntersectRecord &colideRec) {
    bool ishit = false;
    Triangle t;
//	printf("%d cao\n", number);
    for (int i = 0; i < number; i++) {
        t = *(meshTable + i);
        ishit |= t.hit(r, colideRec);
	}
	
	if (ishit) {
		colideRec.material = my_material;
		colideRec.material_type = material_type;
	}

	return ishit;
}

CUDA_FUNC Quadratic::Quadratic(float3 Coefficient, int Type) {
	coefficient = Coefficient;
	type = Type;
	if (Type == Sphere) {
		if (!(coefficient.x == coefficient.y && coefficient.x == coefficient.z)) {
			return;
		}
	}
}
	
CUDA_FUNC bool Quadratic::setHeight(float Height) {
	if (type == Sphere)
		height = Height;
	else
		return false;
	return true;
}
CUDA_FUNC float3 Quadratic::getCenter() const{
	return float3{ transformation.v[0][3],transformation.v[1][3],transformation.v[2][3] };
}
CUDA_FUNC float Quadratic::getRadius() const{
	if (type == Sphere) {
		return (1/coefficient.x);
	}
    return 0.0f;
}


CUDA_FUNC  bool  Quadratic::hit(Ray r, IntersectRecord &colideRec) {
	if (type == Sphere) {
        float3 center = make_float3(0.0f, 0.0f, 0.0f);
        mat4 inv = inverse(transformation);
        float3 Lorigin = inv(r.getOrigin());

        float4 L4dir = inv(make_float4(r.getDir(), 0));
        float3 Ldir = normalize(make_float3(L4dir.x, L4dir.y, L4dir.z));
        float3 oc = Lorigin - center;
        float dotOCD = dot(Ldir, oc);

        if (dotOCD > 0)
            return false;

        float dotOC = dot(oc, oc);
        float discriminant = dotOCD * dotOCD - dotOC + getRadius()*getRadius();
        float t0, t1;
        if (discriminant < 0)
            return false;
        else if (discriminant < FLOAT_EPISLON)
            t0 = t1 = -dotOCD;
        else {
            discriminant = sqrt(discriminant);
            t0 = -dotOCD - discriminant;
            t1 = -dotOCD + discriminant;
            if (t0 < FLOAT_EPISLON)
                t0 = t1;
        }
        //Need to double-check
        //float3 tangent = normalize(make_float3(transformation(make_float4((cross(cross(Ldir*t0 + Lorigin, make_float3(0.0f, 1.0f / , 0.0f)), Ldir*t0 + Lorigin)),0))));
        //float3 normal = normalize(make_float3(transformation(make_float4(Ldir*t0 + Lorigin, 0))));
        //float3 pos = transformation(Ldir*t0 + Lorigin);

        float3 pos = Ldir * t0 + Lorigin;
        float3 normal = normalize(make_float3(transformation(make_float4(pos, 0.0f))));
        float3 tangent;
        if (pos.x == 0.0f && pos.y == 0.0f)
            tangent = make_float3(transformation(make_float4(0.0f, 0.0f, -1.0f, 0.0f)));
        else tangent = cross(
            normal,
            cross(make_float3(transformation(make_float4(UP_VEC - pos, 0.0f))),
                normal)
            );

        tangent = normalize(tangent);
        pos = transformation(pos);

        //if(t0 > 0.0f)
         //   if (dot(normal, r.getDir()) < 0.0f)
         //      printf("%f\n", dot(normal, r.getDir()));

        if (t0 > FLOAT_EPISLON && t0 < colideRec.t)
        {
            //printf("hhh");
            colideRec.material = my_material;
            colideRec.material_type = material_type;
            colideRec.t = t0;
            colideRec.pos = pos;
            colideRec.normal = normal;
            colideRec.tangent = normalize(tangent);
			colideRec.isLight = false;
            return true;
        }
	} 
	else {
		//TODO-Cylinder.
		                               
	}

	return false;
}


CUDA_FUNC float Triangle::area() const
{
    float3 rpos[3];
    for (int i = 0; i < 3; i++)
        rpos[i] = transformation(pos[i]);

    return length(cross(rpos[2] - rpos[0], rpos[1] - rpos[0]));
}

__host__ bool Model::setUpMaterial(material::MATERIAL_TYPE t, Material *mat)
{
    size_t num;
    switch (t)
    {
    case material::LAMBERTIAN:
        num = sizeof(Lambertian);
        break;
    case material::MATERIAL_NUM:
        //break;
    default:
        num = 0;
        return false;
    }
    material_type = t;
    Material tmp = *mat;
    cudaMalloc(&tmp.brdfs, num);
    cudaMemcpy(tmp.brdfs, mat->brdfs, num, cudaMemcpyHostToDevice);
    auto error =  cudaMalloc(&my_material, sizeof(Material));
    error = cudaMemcpy(my_material, &tmp, sizeof(Material), cudaMemcpyHostToDevice);

    return error == cudaSuccess;
}